"""
Z-Image Turbo UINT4 - Fast Image Generation on Mac

Uses the quantized uint4 model (only 3.5GB!) for fast inference on Apple Silicon.
"""

import os
import argparse

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

# Import sdnq FIRST - it registers quantization format with diffusers
import sdnq

from diffusers import ZImagePipeline


def load_pipeline(device="mps"):
    """Load the quantized Z-Image pipeline."""
    print("Loading Z-Image-Turbo UINT4 (3.5GB model)...")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch version: {torch.__version__}")

    pipe = ZImagePipeline.from_pretrained(
        "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    pipe.to(device)

    # Memory optimizations
    # pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        # pipe.enable_vae_slicing()
        print("VAE slicing disabled")

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        # pipe.vae.enable_tiling()
        print("VAE tiling disabled")

    print("Pipeline loaded!")
    return pipe


def generate(
    pipe,
    prompt: str,
    height: int = 512,
    width: int = 512,
    steps: int = 5,
    seed: int = None,
):
    """Generate an image from a prompt."""
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating with seed {seed}...")

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=torch.Generator("mps").manual_seed(seed),
        ).images[0]

    return image, seed


def main():
    parser = argparse.ArgumentParser(description="Generate images with Z-Image Turbo UINT4")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--steps", type=int, default=5, help="Inference steps (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")

    args = parser.parse_args()

    pipe = load_pipeline()
    image, seed = generate(
        pipe,
        args.prompt,
        args.height,
        args.width,
        args.steps,
        args.seed,
    )

    image.save(args.output)
    print(f"Saved to {args.output} (seed: {seed})")


if __name__ == "__main__":
    main()
