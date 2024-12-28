from diffusers import StableDiffusionPipeline
import torch
import re
import random
from PIL import Image, ImageDraw, ImageFont
import os
import logging
from typing import Optional
import time  # Importing time module to fix the undefined variable error

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class MergeBotImageGenerator:
    def __init__(self, font_path: Optional[str] = None):
        self.INAPPROPRIATE_WORDS = [
            'kötü', 'şiddet', 'tehlike', 'ölüm', 'yaralanma', 'korku', 'acı', 'kan', 'savaş', 'ölmek'
        ]
        self.CHILD_FRIENDLY_PREFIXES = [
            "A cute and colorful cartoon of",
            "A friendly and cheerful illustration showing",
            "A magical and playful scene with",
            "A bright and happy drawing of"
        ]
        self.SELECTED_STYLE = "in a colorful cartoon style with soft shapes and exaggerated features"
        self.font_path = font_path or "arial.ttf"  # Default font path
        self.pipe = None
        self.device = self._select_device()
        self._initialize_model()

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info("Using GPU for processing")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Using Apple Metal Performance Shaders")
            return torch.device("mps")
        else:
            logger.warning("Using CPU. Performance may be lower.")
            return torch.device("cpu")

    def _initialize_model(self):
        try:
            model_id = "CompVis/stable-diffusion-v1-4"
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.pipe = None

    def check_content_safety(self, text: str) -> tuple[bool, str]:
        text_lower = text.lower()
        for word in self.INAPPROPRIATE_WORDS:
            if word in text_lower:
                return False, f"'{word}' content is inappropriate. Please use a kinder expression."
        if len(text) > 200:
            return False, "Too long! Please shorten it."
        if len(text) < 3:
            return False, "Too short! Please add more details."
        return True, text

    def generate_child_friendly_prompt(self, original_prompt: str) -> str:
        child_prefix = random.choice(self.CHILD_FRIENDLY_PREFIXES)
        cleaned_prompt = re.sub(r'[^\w\s]', '', original_prompt)
        return f"{child_prefix} {cleaned_prompt} {self.SELECTED_STYLE}"

    def generate_image(self, prompt: str, output_dir: str = 'output_images') -> Optional[str]:
        os.makedirs(output_dir, exist_ok=True)
        is_safe, processed_prompt = self.check_content_safety(prompt)
        if not is_safe:
            logger.warning(f"Unsafe content: {processed_prompt}")
            return None
        if not self.pipe:
            logger.error("Model not loaded.")
            return None

        child_friendly_prompt = self.generate_child_friendly_prompt(processed_prompt)
        try:
            image = self.pipe(
                child_friendly_prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

            output_path = os.path.join(output_dir, f"mergebot_image_{int(time.time())}.png")
            image.save(output_path)
            logger.info(f"Image generated successfully: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    def overlay_text_on_image(self, image_path: str, text: str) -> Optional[str]:
        try:
            # Load the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Load font and set size
            try:
                font = ImageFont.truetype(self.font_path, size=50)  # Larger font size for readability
            except Exception:
                logger.warning("Font not found. Using default system font.")
                font = ImageFont.load_default()

            # Calculate text size and position
            text_width, text_height = draw.textsize(text, font=font)
            position = ((image.width - text_width) // 2, image.height - text_height - 20)  # Centered horizontally, near bottom

            # Add a semi-transparent background for text
            text_bg_padding = 15
            bg_box = [
                position[0] - text_bg_padding,
                position[1] - text_bg_padding,
                position[0] + text_width + text_bg_padding,
                position[1] + text_height + text_bg_padding,
            ]
            draw.rectangle(bg_box, fill=(0, 0, 0, 180))  # Semi-transparent black background

            # Add the text on top of the background
            draw.text(position, text, font=font, fill="white")  # White text for better contrast

            # Save the image with text overlay
            image.save(image_path)
            logger.info(f"Text added to image: {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Error adding text to image: {e}")
            return None

def main():
    generator = MergeBotImageGenerator(
        font_path="C:/Users/halam/Downloads/Baloo_Bhai_2/static/BalooBhai2-Regular.ttf"
    )

    test_prompts = ["Mutlu bir çocuk", "Uçan bir balon", "Renkli bir orman"]
    for prompt in test_prompts:
        image_path = generator.generate_image(prompt)
        if image_path:
            generator.overlay_text_on_image(image_path, prompt)
            logger.info(f"Image created: {image_path}")

if __name__ == "__main__":
    main()