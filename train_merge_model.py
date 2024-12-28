import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, descriptions, tokenizer):
        self.image_paths = image_paths
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Reduced size for CPU
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as img:
                img = img.convert('RGB')
                img = self.transform(img)
            text = self.descriptions[idx]
            text_input = self.tokenizer(text, return_tensors='pt', max_length=77, truncation=True, padding='max_length')
            return img, text_input.input_ids.squeeze(0)
        except Exception as e:
            logger.error(f'Error processing file {self.image_paths[idx]}: {e}')
            raise

def train_model():
    base_folder = 'path/to/your/images'
    image_paths = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if f.endswith('.jpg')]
    descriptions = ['A description for each image'] * len(image_paths)  # Replace with actual descriptions

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    dataset = CustomImageDataset(image_paths, descriptions, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size reduced for CPU

    # Load Stable Diffusion model (use CPU)
    from diffusers import StableDiffusionPipeline
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32  # Use float32 for CPU
    ).to('cpu')

    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=1e-4)
    model.train()

    for epoch in range(1):  # Reduce epochs for CPU
        for img, text in loader:
            img, text = img.to('cpu'), text.to('cpu')  # Use CPU
            optimizer.zero_grad()
            
            # Custom training logic for embeddings/adapters
            outputs = model(img, text_inputs=text)
            loss = outputs.loss  # Replace with specific loss logic
            loss.backward()
            optimizer.step()

            logger.info(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == '__main__':
    train_model()
