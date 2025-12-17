# app.py
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# Download model from a public source (no token needed)
def load_model():
    # Option 1: Use CompVis original SD (publicly available)
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # This model is fully public and doesn't require token
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    
    return pipe

pipe = load_model()

def generate_room_panorama(room_type, style, custom_details, width_multiplier, num_steps):
    # Create panoramic prompt
    base_prompt = f"{room_type} interior design, {style} style"
    if custom_details:
        base_prompt += f", {custom_details}"
    
    full_prompt = f"{base_prompt}, panoramic view, wide angle, professional architecture photography, highly detailed, 8k"
    
    negative_prompt = "blurry, distorted, low quality, cropped, out of frame, duplicate"
    
    # Generate wider image for panoramic effect
    width = 512 * width_multiplier
    height = 512
    
    image = pipe(
        full_prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=7.5
    ).images[0]
    
    return image

# Gradio interface
with gr.Blocks(title="Room Panorama Generator") as demo:
    gr.Markdown("""
    # 🏠 Room Panorama Generator
    Generate cylindrical panoramic views of custom room designs without any API tokens!
    """)
    
    with gr.Row():
        with gr.Column():
            room_type = gr.Dropdown(
                choices=[
                    "living room",
                    "bedroom",
                    "kitchen",
                    "bathroom",
                    "office",
                    "dining room",
                    "home theater",
                    "gym"
                ],
                value="living room",
                label="Room Type"
            )
            
            style = gr.Dropdown(
                choices=[
                    "modern",
                    "minimalist",
                    "industrial",
                    "scandinavian",
                    "traditional",
                    "contemporary",
                    "rustic",
                    "art deco",
                    "mid-century modern",
                    "japanese zen"
                ],
                value="modern",
                label="Design Style"
            )
            
            custom_details = gr.Textbox(
                label="Additional Details",
                placeholder="e.g., large windows, wooden floor, plants, natural light",
                lines=3
            )
            
            width_multiplier = gr.Slider(
                minimum=2,
                maximum=4,
                value=3,
                step=1,
                label="Panorama Width (multiplier)",
                info="Higher = wider panoramic view"
            )
            
            num_steps = gr.Slider(
                minimum=20,
                maximum=50,
                value=30,
                step=5,
                label="Generation Steps",
                info="Higher = better quality but slower"
            )
            
            generate_btn = gr.Button("🎨 Generate Panorama", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(
                label="Generated Panorama",
                type="pil"
            )
            
            gr.Markdown("""
            ### Tips:
            - Width multiplier 3-4 works best for panoramic views
            - Add specific details for better results
            - Generation takes ~30-60 seconds on Colab GPU
            """)
    
    generate_btn.click(
        fn=generate_room_panorama,
        inputs=[room_type, style, custom_details, width_multiplier, num_steps],
        outputs=output_image
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["living room", "modern", "floor-to-ceiling windows, marble floor, designer furniture", 3, 30],
            ["bedroom", "scandinavian", "wooden bed, white walls, minimal decor, cozy", 3, 30],
            ["kitchen", "industrial", "exposed brick, stainless steel appliances, concrete countertops", 4, 35],
        ],
        inputs=[room_type, style, custom_details, width_multiplier, num_steps],
    )

if __name__ == "__main__":
    demo.launch(share=True)