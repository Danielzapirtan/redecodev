import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import json
import sys

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This tells PyTorch not to use GPU
device = torch.device("cpu")

# Load JSON prompts from a separate file
def load_json_prompts():
    # Try to load from external JSON file first
    try:
        if os.path.exists("room_prompts.json"):
            with open("room_prompts.json", "r", encoding="utf-8") as f:
                return f.read();
    except Exception as e:
        print(f"Warning: Could not load external JSON file: {e}")

JSON_PROMPTS = load_json_prompts()

# Download model from a public source (no token needed)
def load_model():
    print("Loading model for CPU...")

    # Use a smaller model that works better on CPU
    model_id = "runwayml/stable-diffusion-v1-5"
    #model_id = "CompVis/stable-diffusion-v1-4"

    # Explicitly set device to CPU and use float32
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use float32 for CPU
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
                use_safetensors=True
                )

        # Force CPU
        pipe = pipe.to(device)

        # Use CPU-optimized scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Enable memory optimizations
        pipe.enable_attention_slicing(1)

        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()

        print("Model loaded successfully on CPU!")
        return pipe

    except Exception as e:
        print(f"Error loading model: {e}")
        # Try alternative model
        try:
            print("Trying alternative model...")
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    low_cpu_mem_usage=True
                    )
            pipe = pipe.to(device)
            pipe.enable_attention_slicing(1)
            print("Alternative model loaded successfully!")
            return pipe
        except Exception as e2:
            print(f"Failed to load alternative model: {e2}")
            raise

def generate_room_panorama_cli(room_type, output_path="generated_room.png"):
    """CLI version for GitHub Actions - FIXED"""
    print(f"Generating panorama for room type: {room_type}")
    print(f"Output path: {output_path}")

    # Load model
    pipe = load_model()

    # Get prompt data
    prompt_data = json.loads(JSON_PROMPTS).get(room_type)
    if not prompt_data:
        available_types = list(json.loads(JSON_PROMPTS).keys())
        raise ValueError(f"Room type '{room_type}' not found. Available types: {available_types}")

    # Generate image with smaller dimensions for CPU
    width = 512 * 2  # Smaller for CPU
    height = 512

    try:
        print("Generating image... (this may take 3-10 minutes on CPU)")
        image = pipe(
            prompt_data,
            negative_prompt="",
            height=height,
            width=width,
            num_inference_steps=50,  # Fewer steps for CPU
            guidance_scale=7.5
        ).images[0]

        # Save image
        image.save(output_path)
        print(f"Image saved to {output_path}")

        # Also save metadata
        metadata = {
            "room_type": room_type,
            "prompt": prompt_data,
            "negative_prompt": "",
            "dimensions": {"width": width, "height": height},
            "parameters": {
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "device": "cpu"
            },
            "room_data": prompt_data
        }

        with open(output_path.replace(".png", "_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_path

    except Exception as e:
        print(f"Error generating image: {e}")
        raise

def update_custom_details(room_type):
    """Update custom_details textbox based on selected room_type"""
    json_data = JSON_PROMPTS.get(room_type, {})
    return json.dumps(json_data, indent=2) if json_data else ""

def generate_room_panorama_gui(room_type, custom_details):
    """GUI version for Gradio interface - FIXED VERSION"""
    # Parse JSON prompt
    try:
        prompt_data = json.loads(custom_details)
        room_desc = prompt_data.get("room_description", {})

        # Extract all relevant information from JSON
        json_style = room_desc.get("style", "Romanian Year-1980-Style")
        theme = room_desc.get("theme", "light cream")
        description = room_desc.get("description", "User's room")
        room_shape = room_desc.get("shape", "rectangular")
        dimensions = room_desc.get("dimensions", [])
        requirement = room_desc.get("requirement", "")
        optimization = room_desc.get("optimization", "")

        # Start building comprehensive prompt
        prompt_parts = []

        # Add basic room description
        prompt_parts.append(f"{description}, {json_style} style")

        # Add theme
        prompt_parts.append(f"{theme} theme")

        # Add room shape and dimensions if available
        if dimensions and len(dimensions) >= 3:
            prompt_parts.append(f"{room_shape} room, {dimensions[0]}m × {dimensions[1]}m × {dimensions[2]}m")

        # Add requirements and optimizations
        if requirement:
            prompt_parts.append(f"requirement: {requirement}")
        if optimization:
            prompt_parts.append(f"optimization: {optimization}")

        # Add ALL furniture items with their constraints
        items = room_desc.get("items", [])
        if items:
            furniture_list = []
            for item in items:
                item_type = item.get("type", "")
                if not item_type:
                    continue

                # Build furniture description
                item_desc = item_type

                # Add alignment/orientation
                align = item.get("align")
                if align:
                    if isinstance(align, list):
                        align_str = " and ".join(align)
                        item_desc += f" positioned at {align_str}"
                    else:
                        item_desc += f" positioned at {align}"

                # Add other properties
                if "style" in item:
                    item_desc += f", {item['style']} style"
                if "accessories" in item:
                    item_desc += f" with {item['accessories']}"
                if "radius" in item:
                    item_desc += f", radius: {item['radius']}m"
                if "sizes" in item and item["sizes"]:
                    sizes = item["sizes"]
                    if len(sizes) >= 2:
                        item_desc += f", {sizes[0]}m × {sizes[1]}m size"
                if "position" in item:
                    item_desc += f", {item['position']} position"
                if "length" in item:
                    item_desc += f", {item['length']}m length"
                if "total_width" in item:
                    item_desc += f", {item['total_width']}m total width"
                if "count" in item and item["count"] > 1:
                    item_desc = f"{item['count']} × {item_desc}"
                if "placement" in item:
                    item_desc += f", {item['placement']} placement"

                furniture_list.append(item_desc)

            if furniture_list:
                prompt_parts.append(f"Furniture: {', '.join(furniture_list)}")

        # Add task from JSON
        task = prompt_data.get("task", "generate a panorama")
        prompt_parts.append(task)

        # Combine all parts
        base_prompt = ". ".join(prompt_parts)

        # Add panorama-specific terms
        full_prompt = f"{base_prompt}, panoramic view, wide angle, professional architecture photography, highly detailed, 8k, realistic interior design"

    except json.JSONDecodeError:
        # Fallback if JSON is invalid
        full_prompt = f"{room_type.replace(' 2', '')} interior design, Romanian Year-1980-Style style, light cream theme, panoramic view, wide angle, professional architecture photography, highly detailed, 8k, realistic interior design"

    # Comprehensive negative prompt
    negative_prompt = "blurry, distorted, low quality, cropped, out of frame, duplicate, large metal windows, modern furniture, contemporary style, empty room, floating objects, unrealistic lighting"

    # Smaller dimensions for CPU
    width = 512
    height = 512

    print(f"Generated prompt: {full_prompt}")
    print(f"Negative prompt: {negative_prompt}")

    try:
        print("Loading model...")
        pipe = load_model()

        print("Generating image...")
        image = pipe(
                full_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=30,  # Slightly more steps for better quality
                guidance_scale=7.5
                ).images[0]

        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Check if running in CLI mode (for GitHub Actions)
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode for GitHub Actions
        if len(sys.argv) < 3:
            print("Usage: python app.py --cli <room_type> [output_path]")
            print(f"Available room types: {list(JSON_PROMPTS.keys())}")
            sys.exit(1)

        room_type = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "generated_room.png"

        try:
            result = generate_room_panorama_cli(room_type, output_path)
            print(f"Success: {result}")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
