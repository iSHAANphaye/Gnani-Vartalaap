"""Hugging Face Spaces Entrypoint for Project Vartalaap (Gradio SDK Compatible).

Integrates the full-duplex FastAPI WebSocket server and the Vintage Nagra Studio
Console seamlessly into a 100% Free Hugging Face Gradio Space (16 GB RAM + 2 vCPU).
"""

from __future__ import annotations

import os
from pathlib import Path
import gradio as gr
import uvicorn

from app.main import app as fastapi_app

# Path to static assets
STATIC_DIR = Path(__file__).resolve().parent / "app" / "static"

with gr.Blocks(title="Vartalaap (वार्तालाप) — Sovereign Indic Voice Engine") as demo:
    gr.HTML(
        """
        <style>
            footer { visibility: hidden !important; }
            .gradio-container { max-width: 1300px !important; margin: 0 auto !important; padding: 0 !important; background: #14110F !important; }
        </style>
        <div style="width: 100%; min-height: 870px; background: #14110F; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.8);">
            <iframe src="/static/index.html" 
                    style="width: 100%; height: 870px; border: none; display: block;" 
                    allow="microphone; autoplay"
                    title="Vartalaap Vintage Nagra Console">
            </iframe>
        </div>
        """
    )

# Mount Gradio interface onto our FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
