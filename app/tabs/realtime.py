"""MOSS-TTS-Realtime tab — low-latency streaming TTS for voice agents."""

import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from model_loader import load_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_realtime_inference(
    text: str,
    reference_audio: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    try:
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        model, processor, dev, sample_rate = load_model("realtime", device, attn_implementation)

        msg_kwargs = {"text": text}
        if reference_audio:
            msg_kwargs["reference"] = [reference_audio]

        conversation = [processor.build_user_message(**msg_kwargs)]
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
            )

        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio_np = messages[0].audio_codes_list[0].cpu().numpy()
            return (sample_rate, audio_np), "✅ Realtime generation completed!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_realtime_tab(args):
    with gr.Column():
        gr.Markdown("### ⚡ MOSS-TTS-Realtime - Low-Latency Voice Agent TTS")
        gr.Markdown(
            "1.7B streaming model optimised for real-time voice agents. "
            "Achieves ~180 ms TTFB after warm-up. "
            "Optionally supply a reference audio to anchor the speaker voice."
        )

        with gr.Row():
            with gr.Column(scale=1):
                rt_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=6,
                    placeholder="Enter text here…",
                )
                rt_reference = gr.Audio(
                    label="Reference Audio (Optional)",
                    type="filepath",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    rt_temp = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Temperature")
                    rt_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="Top P")
                    rt_top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    rt_max_tokens = gr.Slider(256, 4096, value=2048, step=128, label="Max New Tokens")

                rt_generate_btn = gr.Button("⚡ Generate (Realtime)", variant="primary", size="lg")

            with gr.Column(scale=1):
                rt_output = gr.Audio(label="Generated Audio")
                rt_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**About this model:**")
                gr.Markdown(
                    "- Architecture: MossTTSRealtime (1.7B)\n"
                    "- TTFB: ~180 ms (after warm-up)\n"
                    "- Ideal for voice agents paired with LLMs\n"
                    "- Supports multi-turn context via reference audio"
                )

        rt_generate_btn.click(
            fn=lambda *x: run_realtime_inference(*x, args.device, args.attn_implementation),
            inputs=[rt_text, rt_reference, rt_temp, rt_top_p, rt_top_k, rt_max_tokens],
            outputs=[rt_output, rt_status],
        )
