# Current Experiments

This repo is currently organized around inference-time CDC compression profiling for SC26/CDC work.

## Active Question

How fast is the compression-side CDC context path on target GPU systems?

The active timing target is:

```text
diffusion.context_fn(images)
```

This includes latent encoding, hyperprior/BPP computation, quantization, and context feature generation. It excludes diffusion reconstruction and `p_sample_loop()`.

## Active Workflow

1. Run `xparam/profile_compression.py` for one checkpoint label and repeat.
2. Repeat across checkpoint labels and repeats using a Slurm sweep.
3. Aggregate results with `xparam/plot_compression_profile.py`.
4. Compare context time, BPP, estimated raw RGB compression ratio, throughput, and peak GPU memory.

## Active Outputs

Each profile repeat writes:

- `compression_profile_results.csv`
- `compression_profile_report.txt`

Each sweep summary writes:

- `compression_profile_summary.csv`
- `plots/plot_context_time_vs_checkpoint.png`
- `plots/plot_bpp_vs_checkpoint.png`
- `plots/plot_speed_vs_compression_ratio.png`
- `plots/plot_memory_vs_checkpoint.png`
- `plots/plot_context_time_vs_bpp.png`
- `plots/plot_compression_ratio_vs_checkpoint.png`

Default output root:

```text
outputs/compression_profile/<job_id>/
```

## Checkpoints

The active sweep uses six x-param checkpoint families:

- `b0.0032`, LPIPS weight `0.0`
- `b0.0064`, LPIPS weight `0.0`
- `b0.0128`, LPIPS weight `0.0`
- `b0.0512`, LPIPS weight `0.9`
- `b0.1024`, LPIPS weight `0.9`
- `b0.2048`, LPIPS weight `0.9`

Model checkpoints are expected as `.pt` files containing an `ema` state dict. Keep `--lpips_weight` explicit so profiling reports remain reproducible.

## Secondary Workflows

Reconstruction scripts remain available for reference, but they are not the default compression timing workflow:

- `xparam/profile_reconstruction.py`
- `xparam/sweep_steps.py`
- `xparam/plot_results.py`
- `xparam/evaluate_compression.py`

These use `diffusion.compress(...)`, which includes reconstruction through the diffusion sampling loop.
