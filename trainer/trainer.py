import torch
from torch import nn
import wandb

class Trainer():
    def __init__(self,config, dataloaders, generator, discriminator, optim_d, optim_g, criterion, device):
        self.config = config
        self.dataloaders = dataloaders
        self.generator = generator
        self.discriminator = discriminator
        self.optim_d = optim_d
        self.optim_g = optim_g
        self.criterion = criterion
        self.device = device

        self.len_epoch = config["trainer"]["len_epoch"]
        self.noise_size = config["arch"]["Generator"]["args"]["channels"][0]
        self.log_period = config["trainer"]["log_period"]
        self.num_epochs = config["trainer"]["num_epochs"]
    
    def process_batch(self, real, train=True):
        real = real.to(self.device)

        if train:
         self.optim_d.zero_grad()

        real_labels = torch.ones(real.shape[0], 1).to(self.device)
        real_preds = self.discriminator(real)
        real_loss = self.criterion(real_preds, real_labels)

        z = torch.randn(real.shape[0], self.noise_size).to(self.device)
        fake_images = self.generator(z)
        fake_labels = torch.zeros(real.shape[0], 1).to(self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = self.criterion(fake_preds, fake_labels)
        
        d_loss = real_loss + fake_loss
        
        if train:
            d_loss.backward()
            self.optim_d.step()
        
        # Train G
        if train:
            self.optim_g.zero_grad()
        z = torch.randn(real.shape[0], self.noise_size).to(self.device)
        fake_images = self.generator(z)
        fake_preds = self.discriminator(fake_images)
        # fool discriminator
        g_loss = self.criterion(fake_preds, real_labels)
        
        if train:
            g_loss.backward()
            self.optim_g.step()

        return g_loss, real_loss, fake_loss, d_loss, fake_images
        

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        for i, real in enumerate(self.dataloaders["train"]):
            g_loss, real_loss, fake_loss, d_loss, fake_images = self.process_batch(real, True)

            if i % self.log_period == 0:
                wandb.log({"train generator loss": g_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                wandb.log({"train discriminator(real) loss": real_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                wandb.log({"train discriminator(fake) loss": fake_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                wandb.log({"train discriminator loss": d_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                fake_images = fake_images*0.5 + 0.5
                for j in range(5):
                    wandb.log({"train image_{}".format(j): wandb.Image(fake_images[j])}, step=(epoch - 1) * self.len_epoch + i)
    
            if i == self.len_epoch - 1:
                break

    def evaluation(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        for part in self.config["data"]["parts"]:
            total_g_loss = 0
            total_d_loss = 0
            total_fake_loss = 0
            total_real_loss = 0
            for i, real in enumerate(self.dataloaders[part]):
                g_loss, real_loss, fake_loss, d_loss, fake_images = self.process_batch(real, False)
                total_g_loss += g_loss.item() / len(self.dataloaders[part])
                total_d_loss += d_loss.item() / len(self.dataloaders[part])
                total_fake_loss += fake_loss.item() / len(self.dataloaders[part])
                total_real_loss += real_loss.item() / len(self.dataloaders[part])
            

        wandb.log({f"{part} generator loss": total_g_loss}, step=epoch * self.len_epoch)
        wandb.log({f"{part} discriminator(real) loss": total_real_loss}, step=epoch * self.len_epoch)
        wandb.log({f"{part} discriminator(fake) loss": total_fake_loss}, step=epoch * self.len_epoch)
        wandb.log({f"{part} discriminator loss": total_d_loss}, step=epoch * self.len_epoch)
        fake_images = fake_images*0.5 + 0.5
        for j in range(5):
            wandb.log({f"{part} image_{j}": wandb.Image(fake_images[j])})


    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            self.evaluation(epoch)


    
