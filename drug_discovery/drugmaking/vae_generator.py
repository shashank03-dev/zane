"""
VAE-based Delivery System Generator

Uses a Variational Autoencoder to generate novel LNP and polymer compositions.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DeliveryVAE(nn.Module):
    """VAE for generating delivery system compositions."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc21 = nn.Linear(32, latent_dim)  # mean
        self.fc22 = nn.Linear(32, latent_dim)  # logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DeliveryGenerator:
    """Service to generate and optimize delivery systems."""

    def __init__(self, input_dim: int = 4):  # e.g., 4 lipid types for LNP
        self.model = DeliveryVAE(input_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, num_samples: int = 5) -> torch.Tensor:
        """Generate novel compositions."""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.fc21.out_features).to(self.device)
            samples = self.model.decode(z)

        # Normalize to ensure ratios sum to 1
        samples = samples / samples.sum(dim=1, keepdim=True)
        return samples

    def train_on_experimental_data(self, data: torch.Tensor, epochs: int = 100):
        """Fine-tune the VAE on successful experimental compositions."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data.to(self.device))

            # Reconstruction loss + KL divergence
            recon_loss = F.mse_loss(recon_batch, data.to(self.device), reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
