"""
profile_compression.py
----------------------
Profiles CDC learned compression/context-generation performance with a
per-image timing breakdown.

Important: GaussianDiffusion.compress() is an end-to-end wrapper that runs both
context generation and diffusion reconstruction. This script profiles only
diffusion.context_fn(images), the learned encoder / hyperprior / BPP / context
feature generation path used on the compression side.

Usage:
  cd /projects/bfod/$USER/sc26-cdc-deltaai
  python xparam/profile_compression.py \
    --ckpt /path/to/checkpoint.pt \
    --checkpoint_label b0.2048 \
    --img_dir imgs \
    --out_dir /path/to/profile_out \
    --lpips_weight 0.9 \
    --n_images 5 \
    --repeat 1

Outputs:
  compression_profile_results.csv
  compression_profile_report.txt
"""

import argparse
import csv
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torchvision
from ema_pytorch import EMA

# Keep local xparam/modules imports working whether this is launched from the
# repo root or from inside xparam.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modules.compress_modules import ResnetCompressor
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet


UNCOMPRESSED_BPP = 24.0  # RGB, 8 bits per channel
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile CDC context-only compression pipeline")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--checkpoint_label", type=str, default=None, help="Short checkpoint label for reports, e.g. b0.2048")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for profile outputs")
    parser.add_argument("--gamma", type=float, default=0.8, help="Ignored legacy arg; context-only profiling does not sample noise")
    parser.add_argument("--n_denoise_step", type=int, default=65, help="Ignored legacy arg; context-only profiling does not run diffusion sampling")
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
    """CUDA-event timer for GPU-side context-generation latency."""

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
        "num_pixels",
        "megapixels",
        "lpips_weight",
        "model_load_sec",
        "data_load_sec",
        "read_decode_preprocess_cpu_sec",
        "crop_cpu_sec",
        "h2d_sec",
        "context_sec",
        "context_ms_per_megapixel",
        "data_ms_per_megapixel",
        "total_sec",
        "images_per_hour",
        "peak_gpu_mem_mb",
        "bpp",
        "estimated_compressed_bits",
        "estimated_compressed_bytes",
        "raw_rgb_bytes",
        "raw_rgb_to_est_compressed_ratio",
        "compression_ratio",
        "orig_size_bytes",
        "orig_file_to_est_compressed_ratio",
        "context_output_shapes",
        "q_latent_shape",
        "q_hyper_latent_shape",
        "q_latent_numel",
        "q_hyper_latent_numel",
        "context_numel",
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

    print("Profiling CDC context-only compression")
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

        t_read_start = time.perf_counter()
        img_cpu = torchvision.io.read_image(img_path).unsqueeze(0).float() / 255.0
        read_decode_preprocess_cpu_sec = time.perf_counter() - t_read_start

        # Crop on CPU before host-to-device transfer so discarded pixels are not
        # copied to the GPU.
        t_crop_start = time.perf_counter()
        height, width = img_cpu.shape[-2], img_cpu.shape[-1]
        height64 = (height // 64) * 64
        width64 = (width // 64) * 64
        img_cpu = img_cpu[:, :, :height64, :width64]
        height, width = img_cpu.shape[-2], img_cpu.shape[-1]
        crop_cpu_sec = time.perf_counter() - t_crop_start

        t_h2d_start = time.perf_counter()
        tensor = img_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize(device_index)
        h2d_sec = time.perf_counter() - t_h2d_start

        data_load_sec = read_decode_preprocess_cpu_sec + crop_cpu_sec + h2d_sec

        torch.cuda.reset_peak_memory_stats(device_index)

        # CUDA events measure the asynchronous GPU context-generation path accurately.
        timer.start()
        with torch.inference_mode():
            # `diffusion.compress()` includes both the compression/context path and the diffusion
            # reconstruction path. For compression-side profiling, time only `context_fn`, which
            # runs the learned compressor: encoder, hyperprior, quantization, bpp estimation,
            # and context feature generation. The reported bpp is an estimated bitrate from the
            # entropy model, not necessarily an actual entropy-coded file size.
            context_dict = diffusion.context_fn(tensor * 2.0 - 1.0)
        context_sec = timer.stop()
        peak_mem_mb = torch.cuda.max_memory_allocated(device_index) / 1024 ** 2

        context_output_shapes = str([list(t.shape) for t in context_dict["output"]])
        q_latent_shape = str(list(context_dict["q_latent"].shape))
        q_hyper_latent_shape = str(list(context_dict["q_hyper_latent"].shape))
        q_latent_numel = int(context_dict["q_latent"].numel())
        q_hyper_latent_numel = int(context_dict["q_hyper_latent"].numel())
        context_numel = int(sum(t.numel() for t in context_dict["output"]))

        bpp_val = float(context_dict["bpp"].mean())
        num_pixels = int(width * height)
        megapixels = num_pixels / 1e6
        est_compressed_bits = bpp_val * num_pixels
        est_compressed_bytes = est_compressed_bits / 8.0
        raw_rgb_bytes = num_pixels * 3
        raw_rgb_to_est_compressed_ratio = (
            raw_rgb_bytes / est_compressed_bytes if est_compressed_bytes > 0 else float("inf")
        )
        orig_file_to_est_compressed_ratio = (
            orig_bytes / est_compressed_bytes if est_compressed_bytes > 0 else float("inf")
        )
        context_ms_per_megapixel = (
            context_sec * 1000.0 / megapixels if megapixels > 0 else 0.0
        )
        data_ms_per_megapixel = (
            data_load_sec * 1000.0 / megapixels if megapixels > 0 else 0.0
        )
        total_sec = data_load_sec + context_sec
        images_per_hour = 3600.0 / total_sec if total_sec > 0 else 0.0

        row = {
            "checkpoint": checkpoint_label,
            "checkpoint_label": checkpoint_label,
            "ckpt_path": config.ckpt,
            "repeat": config.repeat,
            "image": img_name,
            "width": width,
            "height": height,
            "num_pixels": num_pixels,
            "megapixels": round(megapixels, 4),
            "lpips_weight": config.lpips_weight,
            "model_load_sec": round(model_load_sec, 3),
            "data_load_sec": round(data_load_sec, 3),
            "read_decode_preprocess_cpu_sec": round(read_decode_preprocess_cpu_sec, 3),
            "crop_cpu_sec": round(crop_cpu_sec, 3),
            "h2d_sec": round(h2d_sec, 3),
            "context_sec": round(context_sec, 3),
            "context_ms_per_megapixel": round(context_ms_per_megapixel, 2),
            "data_ms_per_megapixel": round(data_ms_per_megapixel, 2),
            "total_sec": round(total_sec, 3),
            "images_per_hour": round(images_per_hour, 2),
            "peak_gpu_mem_mb": round(peak_mem_mb, 1),
            "bpp": round(bpp_val, 4),
            "estimated_compressed_bits": round(est_compressed_bits, 2),
            "estimated_compressed_bytes": round(est_compressed_bytes, 2),
            "raw_rgb_bytes": raw_rgb_bytes,
            "raw_rgb_to_est_compressed_ratio": round(raw_rgb_to_est_compressed_ratio, 2),
            "compression_ratio": round(raw_rgb_to_est_compressed_ratio, 2),  # deprecated alias
            "orig_size_bytes": orig_bytes,
            "orig_file_to_est_compressed_ratio": round(orig_file_to_est_compressed_ratio, 2),
            "context_output_shapes": context_output_shapes,
            "q_latent_shape": q_latent_shape,
            "q_hyper_latent_shape": q_hyper_latent_shape,
            "q_latent_numel": q_latent_numel,
            "q_hyper_latent_numel": q_hyper_latent_numel,
            "context_numel": context_numel,
            "status": "success",
        }
        results.append(row)

        print(
            f"[{i + 1:3d}/{len(selected)}] {img_name:30s} | "
            f"load {data_load_sec:5.2f}s | "
            f"context {context_sec:6.2f}s | "
            f"total {total_sec:6.2f}s | "
            f"mem {peak_mem_mb:7.1f} MB | "
            f"bpp {bpp_val:.4f} | "
            f"raw/est ratio {raw_rgb_to_est_compressed_ratio:.1f}x"
        )

    successful = [r for r in results if r.get("status") == "success"]
    total_orig_bytes = sum(int(r["orig_size_bytes"]) for r in successful)
    avg_data_load = safe_mean([r["data_load_sec"] for r in successful])
    avg_context = safe_mean([r["context_sec"] for r in successful])
    avg_total = safe_mean([r["total_sec"] for r in successful])
    avg_images_per_hour = safe_mean([r["images_per_hour"] for r in successful])
    avg_mem = safe_mean([r["peak_gpu_mem_mb"] for r in successful])
    avg_bpp = safe_mean([r["bpp"] for r in successful])
    avg_ratio = safe_mean([r["raw_rgb_to_est_compressed_ratio"] for r in successful])

    report_lines = [
        "=" * 72,
        "  CDC CONTEXT-ONLY COMPRESSION PROFILING REPORT",
        "=" * 72,
        "  This script profiles diffusion.context_fn(images).",
        "  This is the learned compression/context-generation stage.",
        "  It includes latent encoding, hyperprior/BPP computation, quantization,",
        "  and context feature generation.",
        "  It excludes diffusion.p_sample_loop().",
        "  It does not measure reconstruction.",
        "  For reconstruction timing, use profile_reconstruction.py.",
        "-" * 72,
        f"  Checkpoint label       : {checkpoint_label}",
        f"  Checkpoint path        : {config.ckpt}",
        f"  Repeat                 : {config.repeat}",
        f"  Images profiled        : {len(successful)}",
        f"  Start index            : {config.start_index}",
        f"  LPIPS weight           : {config.lpips_weight}",
        "-" * 72,
        "  TIMING BREAKDOWN (per-image averages; model load excluded)",
        f"    Model load time       : {model_load_sec:.2f}s  (one-time cost)",
        f"    Data load/preproc     : {avg_data_load:.3f}s",
        f"    Context generation    : {avg_context:.3f}s",
        f"    Total per image       : {avg_total:.3f}s",
        f"    Images/hour           : {avg_images_per_hour:.2f}",
        "-" * 72,
        "  GPU MEMORY",
        f"    After model load      : {mem_after_load_mb:.1f} MB",
        f"    Peak during context   : {avg_mem:.1f} MB (average across images)",
        "-" * 72,
        "  COMPRESSION METRICS",
        f"    Average BPP           : {avg_bpp:.4f}",
        f"    Raw RGB / est. CDC    : {avg_ratio:.2f}x",
        f"    Total original size   : {format_size(total_orig_bytes)} ({total_orig_bytes} bytes)",
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
