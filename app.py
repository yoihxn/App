import streamlit as st
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
import torch
from PIL import Image
import io

# Initialize Accelerator
accelerator = Accelerator()

# Load the Stable Diffusion model with Accelerate
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Ensure model works with Accelerate for multi-GPU or optimized execution
    pipe = pipe.to(accelerator.device)
    return pipe

pipe = load_model()

# Streamlit app title and description
st.title("Deep Learning Project Text-to-Image Generator with Stable Diffusion and Accelerate")
st.markdown("Enter a text prompt to generate an image using Stable Diffusion (optimized with Accelerate).")

# Input field for the user to enter a text prompt
text_input = st.text_input("Enter your text prompt here:", "")

# Sidebar options for image generation (optional)
st.sidebar.title("Image Options")
image_size = st.sidebar.selectbox(
    "Select Image Size",
    ("512x512", "768x768", "1024x1024")  # Stable Diffusion commonly supports 512x512
)

# Convert selected size to width and height
size_mapping = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024)
}
width, height = size_mapping[image_size]

# Button to generate the image
if st.button("Generate Image"):
    if text_input:
        with st.spinner('Generating image...'):
            try:
                # Run inference using Accelerate
                with accelerator.autocast():
                    image = pipe(text_input, height=height, width=width).images[0]

                # Display the generated image
                st.image(image, caption=f"Generated from: {text_input}", use_column_width=True)

                # Allow image download
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a text prompt.")

# Add a footer
st.markdown("---")
st.markdown("Built with Streamlit, Stable Diffusion, and Accelerate")
