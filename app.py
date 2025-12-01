"""
Z-Image Turbo UINT4 - Gradio Web Interface

Fast image generation on Apple Silicon using the quantized uint4 model.
"""

import os
import gc
import time
from datetime import datetime
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import sdnq
import gradio as gr
from diffusers import ZImagePipeline

# Global pipeline and current device
pipe = None
current_device = None
stop_signal = False

def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def cleanup_memory():
    """Force garbage collection and empty cache."""
    global pipe
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_model():
    """Unload the model from memory."""
    global pipe, current_device
    if pipe is not None:
        print("Unloading model...")
        del pipe
        pipe = None
        current_device = None
        cleanup_memory()
        return "Model unloaded!"
    return "Model already unloaded."


def load_pipeline(device="mps"):
    """Load the pipeline (cached globally)."""
    global pipe, current_device

    # Reload if device changed
    if pipe is not None and current_device == device:
        return pipe

    if pipe is not None:
        print(f"Switching device from {current_device} to {device}...")
        unload_model()

    print(f"Loading Z-Image-Turbo UINT4 on {device}...")

    # Use float16 for CUDA and MPS (M-series optimization), float32 for CPU
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16  # Optimization for M-series
    else:
        dtype = torch.float32

    try:
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        pipe.to(device)
        # pipe.enable_attention_slicing()

        if hasattr(pipe, "enable_vae_slicing"):
            # pipe.enable_vae_slicing()
            pass

        if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
            # pipe.vae.enable_tiling()
            pass

        current_device = device
        print(f"Pipeline loaded on {device}!")
        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None


def interrupt_callback(step: int, timestep: int, latents: torch.FloatTensor):
    """Callback to stop generation."""
    global stop_signal
    if stop_signal:
        raise RuntimeError("Generation stopped by user")


def generate_image(prompt, height, width, steps, seed, device):
    """Generate an image from the prompt."""
    global stop_signal
    stop_signal = False
    
    try:
        pipe = load_pipeline(device)
        if pipe is None:
            return None, "Error: Could not load pipeline."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(int(seed))
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(int(seed))

    try:
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=0.0,
                generator=generator,
                callback_on_step_end=interrupt_callback,
            ).images[0]
        
        return image, f"Seed: {seed} | Device: {device}"
    
    except RuntimeError as e:
        if "Generation stopped by user" in str(e):
            return None, "Generation stopped by user."
        raise e
    except Exception as e:
        return None, f"Error during generation: {str(e)}"


def stop_generation():
    """Signal to stop generation."""
    global stop_signal
    stop_signal = True
    return "Stopping..."


def save_generated_image(image, custom_filename):
    """Save the generated image to disk."""
    if image is None:
        return "No image to save!"
    
    try:
        if not custom_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.png"
        else:
            filename = custom_filename
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filename += ".png"
        
        image.save(filename)
        return f"Saved to {os.path.abspath(filename)}"
    except Exception as e:
        return f"Error saving image: {str(e)}"


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
with gr.Blocks(title="Z-Image Turbo UINT4") as demo:
    gr.Markdown("""
    # Z-Image Turbo UINT4
    
    Fast image generation using the quantized 3.5GB model.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )

            with gr.Row():
                height = gr.Slider(256, 1024, value=768, step=64, label="Height")
                width = gr.Slider(256, 1024, value=768, step=64, label="Width")

            with gr.Row():
                steps = gr.Slider(1, 10, value=7, step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            with gr.Row():
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    info="MPS=Mac, CUDA=NVIDIA, CPU=slow"
                )
            
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            
            with gr.Row():
                unload_btn = gr.Button("Unload Model", variant="secondary")
                status_msg = gr.Textbox(label="Status", interactive=False, value="Ready")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")
            
            with gr.Row():
                save_filename = gr.Textbox(
                    label="Save Filename (optional)", 
                    placeholder="my_image.png",
                    scale=2
                )
                save_btn = gr.Button("Save Image", scale=1)
            
            save_status = gr.Textbox(label="Save Status", interactive=False, show_label=False)

    # Examples
    gr.Examples(
        examples=[
            ["A majestic mountain landscape at sunset, dramatic lighting, cinematic"],
            ["Portrait of a young woman, soft studio lighting, professional photography"],
            ["Cyberpunk city street at night, neon lights, rain reflections"],
            ["A cute cat wearing a tiny hat, studio photo, soft lighting"],
            ["Abstract art, vibrant colors, fluid shapes, modern design"],
        ],
        inputs=[prompt],
    )

    # Event handlers
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, steps, seed, device],
        outputs=[output_image, status_msg],
    )
    
    stop_btn.click(
        fn=stop_generation,
        inputs=[],
        outputs=[status_msg],
    )
    
    unload_btn.click(
        fn=unload_model,
        inputs=[],
        outputs=[status_msg],
    )
    
    save_btn.click(
        fn=save_generated_image,
        inputs=[output_image, save_filename],
        outputs=[save_status],
    )

if __name__ == "__main__":
    demo.launch()
