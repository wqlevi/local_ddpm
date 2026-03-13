from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

import wandb


@dataclass
class TrainerConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    log_every: int = 500
    eval_every: int = 500
    num_eval_samples: int = 4
    clip_denoised: bool = True
    compute_fid: bool = True
    fid_feature_dim: int = 2048
    use_wandb: bool = True
    wandb_project: str = "ddpms"
    wandb_run_name: str | None = None
    output_dir: str = "outputs"
    device: str | None = None


class DDPMTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = self._resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.global_step = 0

        self.output_dir = self._resolve_output_dir(self.config.output_dir)
        self.image_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "train.log"
        self.fid_metric = self._build_fid_metric()
        self.wandb_run = self._init_wandb()

        self.betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _resolve_device(preferred: str | None) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_output_dir(output_dir: str) -> Path:
        path = Path(output_dir)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    @staticmethod
    def _extract_batch(batch: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(batch):
            return batch
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if torch.is_tensor(first):
                return first
        raise TypeError(
            "DataLoader batch must be a tensor or tuple/list with tensor first."
        )

    def _build_fid_metric(self) -> FrechetInceptionDistance | None:
        if (
            not self.config.compute_fid
            or self.val_loader is None
            or FrechetInceptionDistance is None
        ):
            return None
        fid_score = FrechetInceptionDistance(
            feature=self.config.fid_feature_dim,
            normalize=False,
        )
        return fid_score.to(
            self.device
        )  # FIXME: TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.

    def _init_wandb(self):
        if not self.config.use_wandb or wandb is None:
            return None
        if wandb.run is not None:
            return wandb.run
        return wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                "timesteps": self.config.timesteps,
                "beta_start": self.config.beta_start,
                "beta_end": self.config.beta_end,
                "log_every": self.config.log_every,
                "eval_every": self.config.eval_every,
                "num_eval_samples": self.config.num_eval_samples,
                "compute_fid": self.config.compute_fid,
                "fid_feature_dim": self.config.fid_feature_dim,
                "device": str(self.device),
                "output_dir": str(self.output_dir),
            },
        )

    def _q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_bar_t.sqrt() * x0 + (1.0 - alpha_bar_t).sqrt() * noise

    @torch.no_grad()
    def _sample_images(
        self, batch_size: int, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        c, h, w = shape
        x = torch.randn(batch_size, c, h, w, device=self.device)
        was_training = self.model.training
        self.model.eval()
        for t_inv in range(self.config.timesteps - 1, -1, -1):
            t = torch.full((batch_size,), t_inv, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1, 1)

            mean = (1.0 / alpha_t.sqrt()) * (
                x - ((1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()) * pred_noise
            )
            if t_inv > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean
        if self.config.clip_denoised:
            x = x.clamp(-1.0, 1.0)
        if was_training:
            self.model.train()
        return x

    @staticmethod
    def _to_uint8_images(images: torch.Tensor) -> torch.Tensor:
        images_uint8 = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        if images_uint8.shape[1] == 1:
            images_uint8 = images_uint8.repeat(1, 3, 1, 1)
        return images_uint8

    def _make_image_grid(self, images: torch.Tensor) -> torch.Tensor:
        # `images` is expected in [-1, 1]; convert to [0, 255].
        img = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
        b, c, h, w = img.shape
        cols = min(4, b)
        rows = (b + cols - 1) // cols
        canvas = torch.zeros((c, rows * h, cols * w), dtype=torch.uint8)
        for i in range(b):
            r = i // cols
            col = i % cols
            canvas[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = img[i]
        return canvas

    def _save_image_grid(self, images: torch.Tensor, tag: str) -> Path:
        canvas = self._make_image_grid(images)

        file_tag = tag.replace("/", "_")
        path = self.image_dir / f"{file_tag}_step_{self.global_step:08d}.png"
        if Image is not None:
            arr = canvas.permute(1, 2, 0).numpy()
            if arr.shape[2] == 1:
                Image.fromarray(arr[:, :, 0], mode="L").save(path)
            else:
                Image.fromarray(arr).save(path)
        else:
            torch.save(canvas, path.with_suffix(".pt"))
        return path

    def _log(
        self,
        message: str = "",
        *,
        metrics: dict[str, float] | None = None,
        loss: float | None = None,
        fid_score: float | None = None,
    ) -> None:
        parts = [message] if message else []
        payload = dict(metrics or {})
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
            payload["loss"] = loss
        if fid_score is not None:
            parts.append(f"fid_score={fid_score:.6f}")
            payload["fid_score"] = fid_score
        line = f"[step={self.global_step}] {' '.join(parts)}"
        print(line)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.wandb_run is not None and payload:
            wandb.log(payload, step=self.global_step)

    def _log_images(self, tag: str, images: torch.Tensor, epoch: int) -> None:
        path = self._save_image_grid(images, tag=tag)
        if self.wandb_run is None:
            return
        if Image is not None and path.suffix == ".png":
            wandb_image = wandb.Image(
                str(path), caption=f"{tag} step={self.global_step}"
            )
        else:
            canvas = self._make_image_grid(images).permute(1, 2, 0).numpy()
            wandb_image = wandb.Image(canvas, caption=f"{tag} step={self.global_step}")
        wandb.log(
            {
                "epoch": epoch,
                tag: wandb_image,
            },
            step=self.global_step,
        )

    @torch.no_grad()
    def _compute_fid(self, max_batches: int | None = None) -> float | None:
        if self.fid_metric is None:
            return None

        self.fid_metric.reset()
        batch_count = 0

        for batch_idx, batch in enumerate(self.val_loader or []):
            real_images = self._extract_batch(batch).to(self.device, non_blocking=True)
            fake_images = self._sample_images(
                batch_size=real_images.shape[0],
                shape=(
                    real_images.shape[1],
                    real_images.shape[2],
                    real_images.shape[3],
                ),
            )
            self.fid_metric.update(self._to_uint8_images(real_images), real=True)
            self.fid_metric.update(self._to_uint8_images(fake_images), real=False)
            batch_count += 1

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

        if batch_count == 0:
            self.fid_metric.reset()
            return None

        fid_score = float(self.fid_metric.compute().item())
        self.fid_metric.reset()
        return fid_score

    def train(self, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            self._train_one_epoch(epoch)

    def _train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        for batch in self.train_loader:
            x0 = self._extract_batch(batch).to(self.device, non_blocking=True)
            t = torch.randint(
                low=0,
                high=self.config.timesteps,
                size=(x0.shape[0],),
                device=self.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x0)
            xt = self._q_sample(x0, t, noise)

            pred_noise = self.model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            if self.global_step % self.config.log_every == 0:
                self._log(
                    f"epoch={epoch} train",
                    metrics={"epoch": float(epoch), "train/loss": loss.item()},
                    loss=loss.item(),
                )
                sample_count = min(self.config.num_eval_samples, x0.shape[0])
                samples = self._sample_images(
                    batch_size=sample_count,
                    shape=(x0.shape[1], x0.shape[2], x0.shape[3]),
                )
                self._log_images("train/sample_images", samples, epoch=epoch)

            if (
                self.val_loader is not None
                and self.global_step % self.config.eval_every == 0
            ):
                val_loss = self.evaluate(max_batches=10)
                fid_score = self._compute_fid(max_batches=10)

                val_metrics = {
                    "epoch": float(epoch),
                    "val/loss": val_loss,
                }
                if fid_score is not None:
                    val_metrics["val/fid_score"] = fid_score
                self._log(
                    f"epoch={epoch} val",
                    metrics=val_metrics,
                    loss=val_loss,
                    fid_score=fid_score,
                )
                self.model.train()

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float:
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(self.val_loader or []):
            x0 = self._extract_batch(batch).to(self.device, non_blocking=True)
            t = torch.randint(
                low=0,
                high=self.config.timesteps,
                size=(x0.shape[0],),
                device=self.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x0)
            xt = self._q_sample(x0, t, noise)

            pred_noise = self.model(xt, t)
            loss = F.mse_loss(pred_noise, noise)
            total_loss += loss.item()
            count += 1

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

        if count == 0:
            if was_training:
                self.model.train()
            return 0.0
        mean_loss = total_loss / count
        if was_training:
            self.model.train()
        return mean_loss
