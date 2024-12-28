from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
import torch
import os
import re
from dotenv import load_dotenv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
if not HF_ACCESS_TOKEN:
    raise ValueError("HF_ACCESS_TOKEN bulunamadı! Lütfen .env dosyasını kontrol edin.")

# Initialize Flask app
app = Flask(__name__)

# Create output directory
OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Inappropriate words list
INAPPROPRIATE_WORDS = {
    'kötü', 'şiddet', 'tehlike', 'ölüm', 'yaralanma'
}

class ImageGenerator:
    def __init__(self):
        self.model_id = "CompVis/stable-diffusion-v1-4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Cihaz kullanımı: {self.device}")
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        try:
            logger.info("Stable Diffusion pipeline başlatılıyor...")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                use_auth_token=HF_ACCESS_TOKEN,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            logger.info("Pipeline başarıyla başlatıldı!")
        except Exception as e:
            logger.error(f"Pipeline başlatılamadı: {e}")
            raise

    def generate(self, prompt, output_path):
        try:
            # Prepend the style instruction to the prompt
            style_instruction = "in a colorful cartoon style with soft shapes and exaggerated features"
            prompt = f"{prompt.strip()} {style_instruction}"
            
            logger.info(f"Resim oluşturuluyor: {prompt}")
            image = self.pipeline(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            return image
        except Exception as e:
            logger.error(f"Resim oluşturma hatası: {e}")
            raise

class TextProcessor:
    @staticmethod
    def check_inappropriate_content(text):
        for word in INAPPROPRIATE_WORDS:
            if re.search(fr'\b{word}\b', text, re.IGNORECASE):
                return False, "Hmm... Bu biraz fazla ciddi oldu! 😬 Hadi, biraz daha eğlenceli bir şeyler yazalım! 😊"
        return True, text

    @staticmethod
    def preprocess_text(text):
        # Basic text cleaning
        text = text.strip()
        # Add more sophisticated Turkish text processing here if needed
        return text

def overlay_text_on_image(image, text):
    try:
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)

        # Try to load a Turkish-compatible font
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (img_with_text.width - text_width) // 2
        y = img_with_text.height - text_height - 20

        # Add white background for better readability
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill='white'
        )

        # Draw text
        draw.text((x, y), text, font=font, fill='black')

        return img_with_text
    except Exception as e:
        logger.error(f"Metin ekleme hatası: {e}")
        return image

# Initialize generators
image_generator = ImageGenerator()
text_processor = TextProcessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        user_input = data.get('sentence', '')

        if not user_input:
            return jsonify({
                'error': 'Ups! Cümle sağlanmadı! 😅 Lütfen tekrar deneyin!'
            }), 400

        # Process text and check content
        processed_text = text_processor.preprocess_text(user_input)
        is_safe, message = text_processor.check_inappropriate_content(processed_text)

        if not is_safe:
            return jsonify({'error': message}), 400

        # Generate unique filename
        filename = f"story_{hash(processed_text)}_{os.urandom(4).hex()}.png"
        output_path = OUTPUT_DIR / filename

        # Generate and save image
        image = image_generator.generate(processed_text, output_path)
        final_image = overlay_text_on_image(image, processed_text)
        final_image.save(output_path)

        logger.info(f"Resim başarıyla oluşturuldu: {filename}")

        return jsonify({
            'message': 'Yay! Resmin başarıyla oluşturuldu! 🎨✨',
            'image_url': f'/output_images/{filename}',
            'processed_text': processed_text
        })

    except Exception as e:
        logger.error(f"Resim oluşturma hatası: {e}")
        return jsonify({
            'error': 'Üzgünüm, bir hata oluştu! Lütfen tekrar deneyin! 😞'
        }), 500

@app.route('/output_images/<filename>')
def serve_image(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except Exception as e:
        logger.error(f"Resim görüntüleme hatası: {e}")
        return jsonify({'error': 'Ups! Resmi bulamadık! 😞'}), 404

# Main function for testing
def main():
    generator = ImageGenerator()
    test_prompts = ["Mutlu bir çocuk", "Uçan bir balon", "Renkli bir orman"]
    for prompt in test_prompts:
        image = generator.generate(prompt, "test_images")
        if image:
            output_path = f"test_images/{prompt.replace(' ', '_')}.png"
            final_image = overlay_text_on_image(image, prompt)
            final_image.save(output_path)
            logger.info(f"Image created: {output_path}")

if __name__ == "__main__":
    logger.info("MergeBot başlatılıyor...")
    main()  # Call the main function for testing
    app.run(debug=True, host="0.0.0.0", port=5002) 