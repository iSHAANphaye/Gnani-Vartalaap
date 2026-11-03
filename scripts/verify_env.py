"""Environment and hardware verification script for Project Vartalaap.

Verifies:
- Python runtime version
- PyTorch version and CUDA support
- GPU device recognition (NVIDIA GeForce RTX 5060)
- VRAM allocation and basic CUDA kernel execution
- Clean imports of core application dependencies
"""

import sys
import importlib
from typing import Dict, Tuple

REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "websockets",
    "pydantic_settings",
    "dotenv",
    "requests",
    "numpy",
    "silero_vad",
    "datasets",
    "soundfile",
    "torch",
    "torchaudio",
    "onnxruntime",
]


def verify_imports() -> Dict[str, Tuple[bool, str]]:
    results = {}
    for mod_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            results[mod_name] = (True, str(version))
        except Exception as e:
            results[mod_name] = (False, str(e))
    return results


def verify_cuda(torch_mod) -> Dict[str, str]:
    info = {
        "cuda_available": str(torch_mod.cuda.is_available()),
        "cuda_version": str(torch_mod.version.cuda),
        "device_count": str(torch_mod.cuda.device_count()),
        "device_name": "None",
        "total_memory_mb": "0",
        "allocated_memory_mb": "0",
        "kernel_test": "FAILED",
    }

    if torch_mod.cuda.is_available():
        device_id = 0
        info["device_name"] = torch_mod.cuda.get_device_name(device_id)
        props = torch_mod.cuda.get_device_properties(device_id)
        info["total_memory_mb"] = f"{props.total_memory / (1024 ** 2):.1f}"

        # Test CUDA tensor allocation and basic kernel execution
        try:
            x = torch_mod.ones((1000, 1000), device="cuda:0")
            y = x @ x
            torch_mod.cuda.synchronize()
            info["allocated_memory_mb"] = f"{torch_mod.cuda.memory_allocated(device_id) / (1024 ** 2):.2f}"
            info["kernel_test"] = f"PASSED (result shape {list(y.shape)})"
            del x, y
            torch_mod.cuda.empty_cache()
        except Exception as e:
            info["kernel_test"] = f"FAILED: {e}"

    return info


def main() -> int:
    print("=" * 65)
    print(" 🎙️  PROJECT VARTALAAP - HARDWARE & ENVIRONMENT VERIFICATION")
    print("=" * 65)

    print(f"\n[1] Python Runtime:")
    print(f"    Version : {sys.version.split()[0]}")
    print(f"    Path    : {sys.executable}")

    print("\n[2] Dependency Module Imports:")
    import_results = verify_imports()
    all_imports_passed = True
    for mod, (ok, ver) in import_results.items():
        status = "✓ OK" if ok else "✗ FAIL"
        if not ok:
            all_imports_passed = False
        print(f"    {status:<8} {mod:<20} {ver}")

    print("\n[3] PyTorch & Hardware Acceleration (RTX 5060):")
    if import_results.get("torch", (False, ""))[0]:
        import torch

        cuda_info = verify_cuda(torch)
        print(f"    PyTorch Version   : {torch.__version__}")
        print(f"    CUDA Available    : {cuda_info['cuda_available']}")
        print(f"    CUDA Version      : {cuda_info['cuda_version']}")
        print(f"    Device Count      : {cuda_info['device_count']}")
        print(f"    Device Name       : {cuda_info['device_name']}")
        print(f"    Total VRAM        : {cuda_info['total_memory_mb']} MB")
        print(f"    Allocated VRAM    : {cuda_info['allocated_memory_mb']} MB")
        print(f"    Kernel Execution  : {cuda_info['kernel_test']}")
        cuda_ok = torch.cuda.is_available() and "PASSED" in cuda_info["kernel_test"]
    else:
        print("    ✗ torch is not installed or import failed.")
        cuda_ok = False

    print("\n" + "=" * 65)
    if all_imports_passed and cuda_ok:
        print(" [RESULT] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY")
        print("=" * 65)
        return 0
    else:
        print(" [RESULT] VERIFICATION FAILED - SEE ABOVE ERRORS")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())
