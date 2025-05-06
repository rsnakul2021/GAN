import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class Generator(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim, channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(768, text_embedding_dim)
        
        # Generator architecture
        self.gen = nn.Sequential(
            # Input: latent_dim + text_embedding_dim
            nn.ConvTranspose2d(latent_dim + text_embedding_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 32 x 32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: channels x 64 x 64
        )

    def forward(self, z, text):
        # Encode text
        text_outputs = self.text_encoder(**text)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Get [CLS] token
        text_features = self.text_projection(text_features)
        
        # Concatenate noise and text features
        z = z.view(z.size(0), -1, 1, 1)
        text_features = text_features.view(text_features.size(0), -1, 1, 1)
        x = torch.cat([z, text_features], dim=1)
        
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, text_embedding_dim, channels=3):
        super(Discriminator, self).__init__()
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(768, text_embedding_dim)
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 512 x 4 x 4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 + text_embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text):
        # Encode text
        text_outputs = self.text_encoder(**text)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Get [CLS] token
        text_features = self.text_projection(text_features)
        
        # Encode image
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Concatenate features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        return self.classifier(combined_features) 