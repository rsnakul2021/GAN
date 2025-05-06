import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import Generator, Discriminator
from transformers import BertTokenizer
import tarfile
import json

class CUBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Read images.txt
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.images.append(os.path.join(root_dir, 'images', img_path))
        
        # Read class labels
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.labels.append(int(class_id) - 1)  # Convert to 0-based indexing
        
        # Read class names
        with open(os.path.join(root_dir, 'classes.txt'), 'r') as f:
            self.class_names = [line.strip().split(' ', 1)[1] for line in f]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        class_name = self.class_names[label]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize text
        text = self.tokenizer(class_name, padding='max_length', max_length=32, 
                            truncation=True, return_tensors='pt')
        # Convert to dictionary format
        text = {k: v.squeeze(0) for k, v in text.items()}
        
        return image, text, label

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = Generator(args.latent_dim, args.text_embedding_dim).to(device)
    discriminator = Discriminator(args.text_embedding_dim).to(device)
    
    # Create optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CUBDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Training loop
    for epoch in range(args.epochs):
        for i, (real_images, text, _) in enumerate(tqdm(dataloader)):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Move text tensors to device
            text = {k: v.to(device) for k, v in text.items()}
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1).to(device)
            d_real_output = discriminator(real_images, text)
            d_real_loss = criterion(d_real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, args.latent_dim).to(device)
            fake_images = generator(z, text)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            d_fake_output = discriminator(fake_images.detach(), text)
            d_fake_loss = criterion(d_fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            fake_labels = torch.ones(batch_size, 1).to(device)
            g_output = discriminator(fake_images, text)
            g_loss = criterion(g_output, fake_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')
        
        # Save checkpoints
        if (epoch + 1) % args.save_interval == 0:
            torch.save(generator.state_dict(), f'checkpoints/generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_{epoch+1}.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/CUB_200_2011')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--text_embedding_dim', type=int, default=256)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    train(args) 