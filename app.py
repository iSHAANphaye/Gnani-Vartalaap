"""Hugging Face Spaces Entrypoint for Project Vartalaap (Gradio SDK Compatible).

Integrates the full-duplex FastAPI WebSocket server and the Vintage Nagra Studio
Console seamlessly into a 100% Free Hugging Face Gradio Space (16 GB RAM + 2 vCPU + ZeroGPU).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np
import gradio as gr
import uvicorn

from app.main import app as fastapi_app

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "app" / "static"
BENCHMARK_REPORT_PATH = ROOT_DIR / "benchmark" / "LIVE_GNANI_BENCHMARK_REPORT.md"

# Load benchmark report markdown if available
if BENCHMARK_REPORT_PATH.exists():
    benchmark_report_md = BENCHMARK_REPORT_PATH.read_text(encoding="utf-8")
else:
    benchmark_report_md = "## Live Benchmark Report\n\nRun `uv run python scripts/run_live_gnani_benchmark_suite.py` to generate report."

ARCHITECTURE_MD = """
## 🏛️ Project Vartalaap Architecture Overview

**Vartalaap (वार्तालाप)** is an ultra-low-latency, dual-stream neuro-symbolic turn-detection engine built natively for Indian multilingual conversation.

### 🌟 Core Architectural Innovations:
1. **Layer 1: Low-Latency Acoustic VAD (<1.0ms)**
   - Continuous 16kHz audio analysis via Silero VAD on CPU with a rolling 8.0-second circular PCM ring buffer.
2. **Layer 2A: On-Device Prosodic Turn Classifier (~6.0ms)**
   - Smart-Turn v3.2 encoder with CUDA Graph acceleration and CPU fallback.
3. **Layer 2B: Real-Time Gnani Prisma Streaming STT & Indic Lexical Parser**
   - WebSocket streaming at `wss://api.vachana.ai/stt/v3/stream` with `x-partials: true` emitting interim tokens every 200ms.
   - Evaluates Indic postpositions (`के`, `இல்`), hanging conjunctions (`और`, `மற்றும்`), cognitive hesitation fillers (`मतलब की`), and terminal closures (`दिया है`).
4. **Two-Stage Cultural Backchannel Soft-Ducking Gate**
   - Stage 1 ducks bot audio to -18 dB within 0.03ms upon user vocalization.
   - Stage 2 evaluates interim tokens (`"haanji"`, `"achha"`, `"theek hai"`) and restores bot playback without interruption.
5. **Continuous Dynamic Expected Turn Silence (ETS) Controller**
   - Automatically expands silence tolerance from 500ms up to 2000ms during mid-sentence thinking pauses, and contracts to ~270ms upon terminal closure.
"""

# ZeroGPU Function Registration
try:
    import spaces
    HAVE_SPACES = True
except ImportError:
    HAVE_SPACES = False

if HAVE_SPACES:
    @spaces.GPU
    def run_zero_gpu_turn_inference():
        """Run accelerated acoustic turn inference on ZeroGPU (NVIDIA A10G)."""
        t0 = time.perf_counter()
        try:
            from app.services.smart_turn import get_smart_turn_engine
            engine = get_smart_turn_engine()
            dummy_audio = np.zeros(16000, dtype=np.float32)
            prob = engine.predict_turn_probability(dummy_audio, sample_rate=16000)
            status = "ONLINE"
        except Exception as exc:
            prob = 0.0
            status = f"Warmup notice: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "hardware": "NVIDIA ZeroGPU (A10G Accelerator)",
            "model": "Smart-Turn v3.2 ONNX / PyTorch",
            "inference_latency_ms": round(latency_ms, 2),
            "turn_probability": round(prob, 4),
            "engine_status": status,
            "indic_sovereign_ai": "Active"
        }
else:
    def run_zero_gpu_turn_inference():
        """CPU fallback acoustic turn inference."""
        t0 = time.perf_counter()
        try:
            from app.services.smart_turn import get_smart_turn_engine
            engine = get_smart_turn_engine()
            dummy_audio = np.zeros(16000, dtype=np.float32)
            prob = engine.predict_turn_probability(dummy_audio, sample_rate=16000)
            status = "ONLINE"
        except Exception as exc:
            prob = 0.0
            status = f"Warmup notice: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "hardware": "CPU (Local / Host Architecture)",
            "model": "Smart-Turn v3.2 CPU ONNX",
            "inference_latency_ms": round(latency_ms, 2),
            "turn_probability": round(prob, 4),
            "engine_status": status,
            "indic_sovereign_ai": "Active"
        }

with gr.Blocks(title="Vartalaap (वार्तालाप) — Sovereign Indic Voice Engine") as demo:
    gr.Markdown(
        """
        # 🎙️ Vartalaap (वार्तालाप)
        ### Sovereign Indic-Aware Hybrid Turn Detection Engine for Conversational Voice Agents
        *Built for **Gnani.ai's "The Great Indian AI Internship Challenge"** • Theme: **Sovereign AI for India***
        """
    )

    with gr.Tabs():
        with gr.TabItem("📻 Vintage Nagra Hardware Console"):
            gr.HTML(
                """
                <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 14px; color: #aaa;">
                        Hardware Simulation: Kudelski Nagra IV-S 15 IPS Studio Tape Deck • Web Audio API & Full-Duplex WebSockets
                    </span>
                    <a href="/static/index.html" target="_blank" 
                       style="padding: 6px 14px; background: #C43B2B; color: #fff; text-decoration: none; border-radius: 6px; font-size: 13px; font-weight: bold;">
                        Open Direct Fullscreen Console ↗
                    </a>
                </div>
                <div style="width: 100%; height: 900px; background: #14110F; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.8);">
                    <iframe src="/static/index.html" 
                            style="width: 100%; height: 100%; border: none; display: block;" 
                            allow="microphone; autoplay"
                            title="Vartalaap Vintage Nagra Console">
                    </iframe>
                </div>
                """
            )

        with gr.TabItem("⚡ ZeroGPU Acoustic Inference"):
            gr.Markdown(
                """
                ### ⚡ On-Demand NVIDIA ZeroGPU Acoustic Inference
                Test on-device Smart-Turn v3.2 deep learning turn-detection directly on Hugging Face's ZeroGPU infrastructure.
                """
            )
            gpu_bench_btn = gr.Button("⚡ Run Smart-Turn v3.2 Inference on ZeroGPU", variant="primary")
            gpu_bench_out = gr.JSON(label="ZeroGPU Hardware & Inference Telemetry")
            gpu_bench_btn.click(fn=run_zero_gpu_turn_inference, inputs=[], outputs=[gpu_bench_out])

        with gr.TabItem("📊 Empirical Live Benchmark Results"):
            gr.Markdown(benchmark_report_md)

        with gr.TabItem("⚙️ System Architecture & Sovereign AI"):
            gr.Markdown(ARCHITECTURE_MD)

# Mount Gradio interface onto our FastAPI application with SSR disabled to prevent Node port collision
app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    port_env = os.environ.get("PORT", "7860")
    try:
        port = int(port_env)
    except ValueError:
        port = 7860
    # Enforce standard Hugging Face ingress port 7860
    if port == 7861:
        port = 7860
    uvicorn.run(app, host="0.0.0.0", port=port)
