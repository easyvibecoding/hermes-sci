"""Hardware detection + MPS-aware prompt hints.

Phase-2 upgrade: MPS is NOT second-class. On Apple Silicon in 2026:
  - 5-10x speedups vs CPU for transformer/CNN inference (at batch >= 8)
  - Unified memory: a 36GB Mac can load models CUDA users need 36GB VRAM for,
    with no PCIe transfer cost
  - Stable for training; just slower than A100 per-iteration

Tier is now partitioned by unified RAM, not chip gen:
  - high     ≥ 36 GB  (M-series Max / Ultra, production-capable)
  - medium   18-35 GB (M-series Pro, research / prototype-capable)
  - limited  <  18 GB (base M-series, toy experiments fit)
  - cpu_only no GPU of any kind
"""
from __future__ import annotations

import dataclasses
import os
import platform
import shutil
import subprocess
from typing import Literal

Tier = Literal["high", "medium", "limited", "cpu_only"]


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    os: str
    arch: str
    has_gpu: bool
    gpu_type: str              # "cuda" / "mps" / "rocm" / ""
    gpu_name: str              # best-effort human-readable
    tier: Tier
    unified_ram_gb: int        # 0 if not unified (CUDA boxes)
    cpu_cores: int


# ── probes ──────────────────────────────────────────────────────────


def _probe_nvidia() -> tuple[bool, str]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return False, ""
    try:
        r = subprocess.run(
            [nvsmi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True, r.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return False, ""


def _probe_mps() -> tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, ""
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=2)
        cpu = r.stdout.strip() or "Apple Silicon"
    except (OSError, subprocess.SubprocessError):
        cpu = "Apple Silicon"
    return True, f"{cpu} (MPS)"


def _sysctl_int(key: str) -> int:
    try:
        r = subprocess.run(["sysctl", "-n", key],
                           capture_output=True, text=True, timeout=2)
        return int(r.stdout.strip() or "0")
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _unified_ram_gb() -> int:
    """Unified-memory size in GB, 0 if not Apple Silicon."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return 0
    mem_bytes = _sysctl_int("hw.memsize")
    return round(mem_bytes / (1024 ** 3)) if mem_bytes else 0


def _cpu_cores() -> int:
    if platform.system() == "Darwin":
        return _sysctl_int("hw.ncpu") or os.cpu_count() or 1
    return os.cpu_count() or 1


# ── tier ────────────────────────────────────────────────────────────


def _estimate_tier(gpu_type: str, gpu_name: str, unified_ram_gb: int) -> Tier:
    if not gpu_type:
        return "cpu_only"
    if gpu_type == "mps":
        if unified_ram_gb >= 36:
            return "high"
        if unified_ram_gb >= 18:
            return "medium"
        return "limited"
    # CUDA: name-based heuristic
    n = gpu_name.lower()
    for high_marker in ("a100", "h100", "h200", "b100", "b200", "l40s",
                        "4090", "3090", "a6000", "a40", "a80"):
        if high_marker in n:
            return "high"
    for mid_marker in ("3080", "4080", "a10", "l4", "v100", "2080 ti",
                       "5090", "5080"):
        if mid_marker in n:
            return "medium"
    return "limited"


def detect() -> HardwareProfile:
    has_cuda, cuda_name = _probe_nvidia()
    if has_cuda:
        gpu_type, gpu_name = "cuda", cuda_name
        ram = 0
    else:
        has_mps, mps_name = _probe_mps()
        if has_mps:
            gpu_type, gpu_name, ram = "mps", mps_name, _unified_ram_gb()
        else:
            gpu_type, gpu_name, ram = "", "", 0
    return HardwareProfile(
        os=platform.system(),
        arch=platform.machine(),
        has_gpu=bool(gpu_type),
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        tier=_estimate_tier(gpu_type, gpu_name, ram),
        unified_ram_gb=ram,
        cpu_cores=_cpu_cores(),
    )


# ── MPS env setup ───────────────────────────────────────────────────


def apply_mps_env(*, high_watermark: float | None = None,
                  low_watermark: float | None = None) -> dict[str, str]:
    """Export the env vars you want set BEFORE `import torch` on Apple Silicon.

    - Always sets PYTORCH_ENABLE_MPS_FALLBACK=1 (so unimplemented ops fall
      back to CPU instead of raising).
    - Optionally sets the two watermark ratios. Defaults leave them unset;
      only pass values when you know why.

    Returns the dict of env vars that were set (for logging).
    """
    set_env: dict[str, str] = {}
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_env["PYTORCH_ENABLE_MPS_FALLBACK"] = os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]
    if high_watermark is not None:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_watermark)
        set_env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_watermark)
    if low_watermark is not None:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(low_watermark)
        set_env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(low_watermark)
    return set_env


# ── Prompt hints ────────────────────────────────────────────────────


def _mps_scale_guidance(ram_gb: int) -> str:
    """Sensible experiment-scale ceiling so LLM's compute claims stay honest.
    ~60% of unified RAM is a safe upper bound for model + data + scratch."""
    model_ceiling_gb = max(1, int(ram_gb * 0.6))
    return (
        f"unified memory {ram_gb} GB "
        f"(practical model+data ceiling ≈ {model_ceiling_gb} GB)"
    )


def hint_for_prompt(hw: HardwareProfile | None = None) -> str:
    """Short hint injected into experiment-section prompts so the LLM's
    compute claims stay within the author's realistic budget."""
    hw = hw or detect()

    if hw.gpu_type == "cuda":
        if hw.tier == "high":
            return (
                f"(Author's compute: NVIDIA {hw.gpu_name}. Describe experiments "
                f"at full research scale. float16/bfloat16 mixed-precision OK.)"
            )
        if hw.tier == "medium":
            return (
                f"(Author's compute: NVIDIA {hw.gpu_name}. Mid-scale experiments "
                f"(model up to a few hundred million params). Use bfloat16 AMP.)"
            )
        return (
            f"(Author's compute: NVIDIA {hw.gpu_name}. Keep experiments modest: "
            f"model <100M params, batch ≤32, epochs ≤20.)"
        )

    if hw.gpu_type == "mps":
        guidance = _mps_scale_guidance(hw.unified_ram_gb)
        common = (
            f"MPS usage tips: use `device=torch.device('mps')`, keep dtype "
            f"float32 (MPS float16/AMP is immature and often NOT faster), "
            f"batch size ≥8 (small batches lose to dispatch overhead), set "
            f"`PYTORCH_ENABLE_MPS_FALLBACK=1` so unimplemented ops fall back. "
            f"No distributed/multi-GPU on MPS."
        )
        if hw.tier == "high":
            return (
                f"(Author's compute: {hw.gpu_name}, {guidance}. Can run "
                f"production-scale fine-tuning or inference for 1-10B param "
                f"models. {common})"
            )
        if hw.tier == "medium":
            return (
                f"(Author's compute: {hw.gpu_name}, {guidance}. Good for "
                f"research prototypes: models up to a few hundred million "
                f"params, fine-tuning small LLMs, diffusion inference. {common})"
            )
        # limited
        return (
            f"(Author's compute: {hw.gpu_name}, {guidance}. Keep experiments "
            f"small: models <100M params, batch 8-32, epochs ≤20, datasets "
            f"≤50K samples. Do NOT claim A100/H100 or multi-GPU training. "
            f"MPS is still faster than CPU for transformer/CNN inference. {common})"
        )

    return (
        f"(Author's compute: CPU only on {hw.os}/{hw.arch}, {hw.cpu_cores} cores. "
        f"Do NOT claim GPU training or large models. Use small-scale "
        f"analytical experiments or classical ML baselines.)"
    )
