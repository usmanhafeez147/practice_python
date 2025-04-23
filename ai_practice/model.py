import os
import io
import base64
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from datasets import load_dataset

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------
# 1. Dataset and Preprocessing
# -------------------------------

# Define a PyTorch Dataset wrapper for the Hugging Face dataset
class CelebADataset(Dataset):
    def __init__(self, split="train", transform=None, limit=None):
        # Load the CelebA dataset from Hugging Face
        # (Make sure you have an internet connection for the first download)
        self.dataset = load_dataset("celeb_a", split=split)
        self.transform = transform

        # For demonstration, you may limit the number of samples
        if limit is not None:
            self.dataset = self.dataset.select(range(limit))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # The dataset returns a dict with keys "image" and "attributes"
        # "attributes" is a dict; we use the "Male" attribute as label (1 for male, 0 for female)
        image = sample["image"]
        label = sample["attributes"]["Male"]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Define image transforms: resize to 64x64, convert to tensor, and normalize to [-1,1]
img_size = 64
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------------
# 2. Define the Conditional GAN Models
# -------------------------------

latent_dim = 100   # Size of the noise vector
n_classes = 2      # 0: female, 1: male
img_channels = 3   # RGB images
feature_maps = 64  # Base number of filters

# Generator: takes a noise vector and a label and outputs an image
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.net = nn.Sequential(
            # Input: (latent_dim*2) x 1 x 1
            nn.ConvTranspose2d(latent_dim * 2, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output: (img_channels) x 64 x 64
        )
    
    def forward(self, noise, labels):
        # Embed the labels and concatenate with the noise vector
        label_embedding = self.label_emb(labels)  # (batch_size, latent_dim)
        x = torch.cat([noise, label_embedding], dim=1)  # (batch_size, latent_dim*2)
        # Reshape for convolution: (batch_size, latent_dim*2, 1, 1)
        x = x.unsqueeze(2).unsqueeze(3)
        img = self.net(x)
        return img

# Discriminator: takes an image and a label and outputs a probability (real/fake)
class Discriminator(nn.Module):
    def __init__(self, n_classes, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, img_size * img_size)
        self.net = nn.Sequential(
            # Input: (img_channels + 1) x 64 x 64  (we will append a label channel)
            nn.Conv2d(img_channels + 1, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps*2 x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps*4 x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps*8 x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # Output: scalar probability
        )
    
    def forward(self, img, labels):
        batch_size = img.size(0)
        # Embed labels and reshape to a single channel image of size 64x64
        label_embedding = self.label_emb(labels)  # shape: (batch_size, 64*64)
        label_embedding = label_embedding.view(batch_size, 1, img_size, img_size)
        # Concatenate image and label along channel dimension
        x = torch.cat([img, label_embedding], dim=1)
        validity = self.net(x)
        return validity.view(-1)

# -------------------------------
# 3. Training the GAN
# -------------------------------

def train_gan(num_epochs=1, batch_size=32, dataset_limit=1000, device="cpu"):
    # Create dataset and dataloader
    dataset = CelebADataset(split="train", transform=transform, limit=dataset_limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize models
    generator = Generator(latent_dim, n_classes, img_channels, feature_maps).to(device)
    discriminator = Discriminator(n_classes, img_channels, feature_maps).to(device)

    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Labels for real and fake images
    real_label = 1.0
    fake_label = 0.0

    print("Starting GAN training...")
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size_curr = imgs.size(0)
            imgs = imgs.to(device)
            labels = labels.to(device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            valid = torch.full((batch_size_curr,), real_label, dtype=torch.float, device=device)
            output_real = discriminator(imgs, labels)
            loss_real = adversarial_loss(output_real, valid)

            # Fake images
            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            # Random labels for generation
            gen_labels = torch.randint(0, n_classes, (batch_size_curr,), device=device)
            fake_imgs = generator(noise, gen_labels)
            fake = torch.full((batch_size_curr,), fake_label, dtype=torch.float, device=device)
            output_fake = discriminator(fake_imgs.detach(), gen_labels)
            loss_fake = adversarial_loss(output_fake, fake)

            # Total discriminator loss
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # We want generator to trick discriminator: labels are real
            valid.fill_(real_label)
            output = discriminator(fake_imgs, gen_labels)
            loss_G = adversarial_loss(output, valid)
            loss_G.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Save the generator model after training
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/generator.pth")
    print("Training finished. Generator model saved to models/generator.pth")
    return generator

# -------------------------------
# 4. FastAPI Application for Inference
# -------------------------------

# Create a FastAPI app instance
app = FastAPI()

# Load the generator model (if exists) and set to evaluation mode.
# If not found, you might want to train it by calling train_gan().
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim, n_classes, img_channels, feature_maps).to(device)
model_path = "models/generator.pth"
if os.path.exists(model_path):
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("Loaded generator model for inference.")
else:
    # Optionally, you can trigger training here if no model exists.
    print("No pre-trained model found. Please run with the '--train' flag to train the model.")
    
# Define a Pydantic model for POST requests if needed.
class GenerationRequest(BaseModel):
    gender: str  # "male" or "female"

def tensor_to_base64_img(tensor_img):
    """Convert a tensor image (normalized [-1,1]) to a base64-encoded PNG image."""
    # Denormalize and convert to PIL image
    tensor_img = (tensor_img * 0.5 + 0.5).clamp(0, 1)
    grid = transforms.ToPILImage()(tensor_img.squeeze(0).cpu())
    buffered = io.BytesIO()
    grid.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.get("/generate_image")
def generate_image(gender: str):
    """
    Generate an image conditioned on the gender.
    Query parameter:
      - gender: "male" or "female"
    Returns a base64-encoded PNG image.
    """
    if gender.lower() not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
    
    # Convert gender to label: 1 for male, 0 for female
    label = torch.tensor([1], device=device) if gender.lower() == "male" else torch.tensor([0], device=device)
    noise = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        fake_img = generator(noise, label)
    img_base64 = tensor_to_base64_img(fake_img)
    return {"generated_image": img_base64}

# -------------------------------
# 5. Main entry point
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the GAN model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of training samples (for demo)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    args = parser.parse_args()

    if args.train:
        # Train and save the generator
        generator = train_gan(num_epochs=args.epochs, batch_size=args.batch_size,
                              dataset_limit=args.limit, device=device)
        generator.eval()

    # Start the FastAPI server (use: uvicorn image_generator:app --reload)
    import uvicorn
    print("Starting API server...")
    uvicorn.run(app, host=args.host, port=args.port)
