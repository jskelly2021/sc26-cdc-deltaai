"""
profile_compression.py
----------------------
Profiles CDC compression / encoding performance with a per-image timing
breakdown that mirrors the reconstruction profiling table.

The model architecture and diffusion.compress(...) call are intentionally kept
the same as evaluate_compression.py. This script only adds profiling metadata:
data-load timing, CUDA-event compression timing, PNG write timing, peak GPU
memory, checkpoint/repeat labels, and a compact report.

Usage:
  python profile_compression.py \
    --ckpt /path/to/checkpoint.pt \
    --checkpoint_label b0.2048 \
    --img_dir /path/to/drone_imgs \
    --out_dir /path/to/profile_out \
    --lpips_weight 0.9 \
    --n_images 5 \
    --repeat 1

Outputs:
  compression_profile_results.csv
  compression_profile_report.txt
  *_recon.png reconstructed images
"""

import argparse
import csv
import os
import pathlib
import time

import numpy as np
import torch
import torchvision
from ema_pytorch import EMA

from modules.compress_modules import ResnetCompressor
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet


UNCOMPRESSED_BPP = 24.0  # RGB, 8 bits per channel
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile CDC compression / encoding pipeline")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--checkpoint_label", type=str, default=None, help="Short checkpoint label for reports, e.g. b0.2048")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for reconstructions and profile outputs")
    parser.add_argument("--gamma", type=float, default=0.8, help="Noise init scale")
    parser.add_argument("--n_denoise_step", type=int, default=65, help="Number of diffusion denoising steps")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--lpips_weight", type=float, required=True, help="LPIPS auxiliary loss weight used during training")
    parser.add_argument("--n_images", type=int, default=5, help="Number of images to profile")
    parser.add_argument("--start_index", type=int, default=0, help="Zero-based image start index within sorted image list")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index recorded in the output CSV")
    return parser.parse_args()


def get_cuda_device(device_index: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script inside a GPU allocation.")
    visible_devices = torch.cuda.device_count()
    if device_index < 0 or device_index >= visible_devices:
        raise RuntimeError(
            f"Requested CUDA device {device_index}, but PyTorch sees {visible_devices} visible device(s)."
        )
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index), device_index


class CudaTimer:
    """CUDA-event timer for GPU-side compression latency."""

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize(self.device_index)
        return self.start_event.elapsed_time(self.end_event) / 1000.0


def load_model(ckpt_path: str, device: torch.device, lpips_weight: float):
    """
    Build and load the CDC model using the same architecture/settings as
    evaluate_compression.py.
    """
    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=64,
        dim_mults=[1, 2, 3, 4, 5, 6],
        context_dim_mults=[1, 2, 3, 4],
        embd_type="01",
    )
    context_model = ResnetCompressor(
        dim=64,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2, 1],
        hyper_dims_mults=[4, 4, 4],
        channels=3,
        out_channels=64,
    )
    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=lpips_weight,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(ckpt_path, map_location=lambda s, _: s)
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"])
    diffusion = ema.ema_model
    diffusion.to(device)
    diffusion.eval()
    return diffusion


def format_size(bytes_val: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def fieldnames():
    return [
        "checkpoint",
        "checkpoint_label",
        "ckpt_path",
        "repeat",
        "image",
        "width",
        "height",
        "n_denoise_step",
        "gamma",
        "lpips_weight",
        "model_load_sec",
        "data_load_sec",
        "compress_sec",
        "write_sec",
        "total_sec",
        "images_per_hour",
        "peak_gpu_mem_mb",
        "bpp",
        "compression_ratio",
        "orig_size_bytes",
        "recon_size_bytes",
        "status",
    ]


def write_csv(csv_path: pathlib.Path, results):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(results)


def main():
    config = parse_args()
    device, device_index = get_cuda_device(config.device)
    checkpoint_label = config.checkpoint_label or pathlib.Path(config.ckpt).stem

    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Profiling CDC compression")
    print(f"Checkpoint : {config.ckpt}")
    print(f"Label      : {checkpoint_label}")
    print(f"Repeat     : {config.repeat}")
    print(f"Output dir : {out_dir}\n")

    all_imgs = sorted([
        f for f in os.listdir(config.img_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])
    selected = all_imgs[config.start_index : config.start_index + config.n_images]
    print(
        f"Found {len(all_imgs)} images, profiling {len(selected)} "
        f"starting from index {config.start_index}."
    )

    print("Loading model...")
    torch.cuda.reset_peak_memory_stats(device_index)
    t_load_start = time.perf_counter()
    diffusion = load_model(config.ckpt, device, config.lpips_weight)
    torch.cuda.synchronize(device_index)
    model_load_sec = time.perf_counter() - t_load_start
    mem_after_load_mb = torch.cuda.memory_allocated(device_index) / 1024 ** 2
    print(f"Model loaded in {model_load_sec:.2f}s | GPU memory after load: {mem_after_load_mb:.1f} MB\n")

    results = []
    timer = CudaTimer(device_index)

    for i, img_name in enumerate(selected):
        img_path = os.path.join(config.img_dir, img_name)
        orig_bytes = os.path.getsize(img_path)

        # Wall-clock timing captures CPU image decode, host-to-device transfer,
        # and cropping/preprocessing work.
        t_data_start = time.perf_counter()
        tensor = torchvision.io.read_image(img_path).unsqueeze(0).float().to(device) / 255.0
        height, width = tensor.shape[-2], tensor.shape[-1]
        height64 = (height // 64) * 64
        width64 = (width // 64) * 64
        tensor = tensor[:, :, :height64, :width64]
        height, width = tensor.shape[-2], tensor.shape[-1]
        torch.cuda.synchronize(device_index)
        data_load_sec = time.perf_counter() - t_data_start

        torch.cuda.reset_peak_memory_stats(device_index)

        # CUDA events measure the asynchronous GPU compression path accurately.
        timer.start()
        with torch.no_grad():
            compressed, bpp = diffusion.compress(
                tensor * 2.0 - 1.0,
                sample_steps=config.n_denoise_step,
                bpp_return_mean=True,
                init=torch.randn_like(tensor) * config.gamma,
            )
        compress_sec = timer.stop()
        peak_mem_mb = torch.cuda.max_memory_allocated(device_index) / 1024 ** 2

        # Wall-clock timing is appropriate for clamp/rescale, device-to-host
        # transfer, and PNG writing.
        t_write_start = time.perf_counter()
        compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
        out_path = out_dir / f"{pathlib.Path(img_name).stem}_repeat{config.repeat:02d}_recon.png"
        torchvision.utils.save_image(compressed.cpu(), str(out_path))
        write_sec = time.perf_counter() - t_write_start

        recon_bytes = os.path.getsize(str(out_path))
        bpp_val = float(bpp)
        compression_ratio = UNCOMPRESSED_BPP / bpp_val if bpp_val > 0 else float("inf")
        total_sec = data_load_sec + compress_sec + write_sec
        images_per_hour = 3600.0 / total_sec if total_sec > 0 else 0.0

        row = {
            "checkpoint": checkpoint_label,
            "checkpoint_label": checkpoint_label,
            "ckpt_path": config.ckpt,
            "repeat": config.repeat,
            "image": img_name,
            "width": width,
            "height": height,
            "n_denoise_step": config.n_denoise_step,
            "gamma": config.gamma,
            "lpips_weight": config.lpips_weight,
            "model_load_sec": round(model_load_sec, 3),
            "data_load_sec": round(data_load_sec, 3),
            "compress_sec": round(compress_sec, 3),
            "write_sec": round(write_sec, 3),
            "total_sec": round(total_sec, 3),
            "images_per_hour": round(images_per_hour, 2),
            "peak_gpu_mem_mb": round(peak_mem_mb, 1),
            "bpp": round(bpp_val, 4),
            "compression_ratio": round(compression_ratio, 2),
            "orig_size_bytes": orig_bytes,
            "recon_size_bytes": recon_bytes,
            "status": "success",
        }
        results.append(row)

        print(
            f"[{i + 1:3d}/{len(selected)}] {img_name:30s} | "
            f"load {data_load_sec:5.2f}s | "
            f"compress {compress_sec:6.2f}s | "
            f"write {write_sec:5.2f}s | "
            f"total {total_sec:6.2f}s | "
            f"mem {peak_mem_mb:7.1f} MB | "
            f"bpp {bpp_val:.4f} | "
            f"ratio {compression_ratio:.1f}x"
        )

    successful = [r for r in results if r.get("status") == "success"]
    total_orig_bytes = sum(int(r["orig_size_bytes"]) for r in successful)
    total_recon_bytes = sum(int(r["recon_size_bytes"]) for r in successful)
    avg_data_load = safe_mean([r["data_load_sec"] for r in successful])
    avg_compress = safe_mean([r["compress_sec"] for r in successful])
    avg_write = safe_mean([r["write_sec"] for r in successful])
    avg_total = safe_mean([r["total_sec"] for r in successful])
    avg_images_per_hour = safe_mean([r["images_per_hour"] for r in successful])
    avg_mem = safe_mean([r["peak_gpu_mem_mb"] for r in successful])
    avg_bpp = safe_mean([r["bpp"] for r in successful])
    avg_ratio = safe_mean([r["compression_ratio"] for r in successful])
    file_size_ratio = total_orig_bytes / total_recon_bytes if total_recon_bytes > 0 else 0.0

    report_lines = [
        "=" * 72,
        "  CDC COMPRESSION PROFILING REPORT",
        "=" * 72,
        f"  Checkpoint label       : {checkpoint_label}",
        f"  Checkpoint path        : {config.ckpt}",
        f"  Repeat                 : {config.repeat}",
        f"  Images profiled        : {len(successful)}",
        f"  Start index            : {config.start_index}",
        f"  Denoising steps        : {config.n_denoise_step}",
        f"  Gamma                  : {config.gamma}",
        f"  LPIPS weight           : {config.lpips_weight}",
        "-" * 72,
        "  TIMING BREAKDOWN (per-image averages; model load excluded)",
        f"    Model load time       : {model_load_sec:.2f}s  (one-time cost)",
        f"    Data load/preproc     : {avg_data_load:.3f}s",
        f"    Compression           : {avg_compress:.3f}s",
        f"    PNG write/postproc    : {avg_write:.3f}s",
        f"    Total per image       : {avg_total:.3f}s",
        f"    Images/hour           : {avg_images_per_hour:.2f}",
        "-" * 72,
        "  GPU MEMORY",
        f"    After model load      : {mem_after_load_mb:.1f} MB",
        f"    Peak during compress  : {avg_mem:.1f} MB (average across images)",
        "-" * 72,
        "  COMPRESSION METRICS",
        f"    Average BPP           : {avg_bpp:.4f}",
        f"    Compression ratio     : {avg_ratio:.2f}x vs uncompressed RGB",
        f"    Total original size   : {format_size(total_orig_bytes)} ({total_orig_bytes} bytes)",
        f"    Total reconstructed   : {format_size(total_recon_bytes)} ({total_recon_bytes} bytes)",
        f"    File size ratio       : {file_size_ratio:.2f}x  (orig files / recon PNG)",
        "=" * 72,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = out_dir / "compression_profile_report.txt"
    report_path.write_text(report_text + "\n")
    print(f"\nReport saved -> {report_path}")

    csv_path = out_dir / "compression_profile_results.csv"
    write_csv(csv_path, results)
    print(f"CSV saved    -> {csv_path}")


if __name__ == "__main__":
    main()
