"""MOSS-TTSD tab — multi-speaker dialogue generation."""

import re
import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from model_loader import load_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_ttsd_inference(
    script_text: str,
    num_speakers: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    try:
        if not script_text or not script_text.strip():
            return None, "❌ Error: Please enter dialogue script"

        used_speakers = set(int(m) for m in re.findall(r'\[S(\d+)\]', script_text))
        if not used_speakers:
            return None, "❌ Error: Dialogue must include speaker tags like [S1], [S2], …"
        if max(used_speakers) > num_speakers:
            return (
                None,
                f"❌ Error: Script uses [S{max(used_speakers)}] but only {num_speakers} speaker(s) configured",
            )

        model, processor, dev, sample_rate = load_model("ttsd", device, attn_implementation)

        conversation = [processor.build_user_message(text=script_text)]
        batch = processor(conversation, mode="generation", num_speakers=num_speakers)
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
                audio_repetition_penalty=repetition_penalty,
            )

        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio_np = messages[0].audio_codes_list[0].cpu().numpy()
            return (sample_rate, audio_np), "✅ Dialogue generation completed!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ttsd_tab(args):
    with gr.Column():
        gr.Markdown("### 💬 MOSS-TTSD - Multi-Speaker Dialogue Generation")
        gr.Markdown("Generate expressive multi-speaker dialogues from scripts.")

        with gr.Row():
            with gr.Column(scale=1):
                ttsd_script = gr.Textbox(
                    label="Dialogue Script",
                    lines=10,
                    placeholder=(
                        "[S1] Hello, how are you doing today?\n"
                        "[S2] I'm doing great, thanks for asking!\n"
                        "[S1] That's wonderful to hear."
                    ),
                    info="Use [S1], [S2], … tags to label each speaker's turn.",
                )
                ttsd_num_speakers = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Number of Speakers",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    ttsd_temp = gr.Slider(0.1, 3.0, value=1.1, step=0.05, label="Temperature")
                    ttsd_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="Top P")
                    ttsd_top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    ttsd_rep_penalty = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    ttsd_max_tokens = gr.Slider(256, 8192, value=2000, step=128, label="Max New Tokens")

                ttsd_generate_btn = gr.Button("🎭 Generate Dialogue", variant="primary", size="lg")

            with gr.Column(scale=1):
                ttsd_output = gr.Audio(label="Generated Dialogue")
                ttsd_status = gr.Textbox(label="Status", lines=3, interactive=False)

        ttsd_generate_btn.click(
            fn=lambda *x: run_ttsd_inference(*x, args.device, args.attn_implementation),
            inputs=[ttsd_script, ttsd_num_speakers, ttsd_temp, ttsd_top_p, ttsd_top_k, ttsd_rep_penalty, ttsd_max_tokens],
            outputs=[ttsd_output, ttsd_status],
        )
