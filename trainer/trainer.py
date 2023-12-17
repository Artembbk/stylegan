import torch
from torch import nn
import wandb
import piq
from tqdm import tqdm
from dataset import FakeDataset
from torch.utils.data import DataLoader

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
        real = real["images"].to(self.device)

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

        if not train:
            ssim = piq.ssim(fake_images*0.5 + 0.5, real*0.5 + 0.5, data_range=1.)
            return g_loss, real_loss, fake_loss, d_loss, ssim, fake_images
        return g_loss, real_loss, fake_loss, d_loss
        

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        for i, real in tqdm(enumerate(self.dataloaders["train"]), total=self.len_epoch, desc="Train"):
            g_loss, real_loss, fake_loss, d_loss = self.process_batch(real, True)

            if i % self.log_period == 0:
                wandb.log({"train generator loss": g_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                wandb.log({"train discriminator(real) loss": real_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                wandb.log({"train discriminator(fake) loss": fake_loss.item()}, step=(epoch - 1) * self.len_epoch + i)
                wandb.log({"train discriminator loss": d_loss.item()}, step=(epoch - 1) * self.len_epoch + i)

            if i == self.len_epoch - 1:
                break

    def evaluation(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        all_fake_images = []
        for part in self.config["data"]["parts"]:
            if part == "train":
                continue
            total_g_loss = 0
            total_d_loss = 0
            total_fake_loss = 0
            total_real_loss = 0
            total_ssim = 0
            for i, real in tqdm(enumerate(self.dataloaders[part]), total=len(self.dataloaders[part]), desc=part):
                g_loss, real_loss, fake_loss, d_loss, ssim, fake_images = self.process_batch(real, False)
                all_fake_images.append(fake_images.detach().cpu())
                total_g_loss += g_loss.item() / len(self.dataloaders[part])
                total_d_loss += d_loss.item() / len(self.dataloaders[part])
                total_fake_loss += fake_loss.item() / len(self.dataloaders[part])
                total_real_loss += real_loss.item() / len(self.dataloaders[part])
                total_ssim += ssim.item() / len(self.dataloaders[part])

            all_fake_images = torch.cat(all_fake_images)
            fake_dataset = FakeDataset(all_fake_images)
            fake_dataloader = DataLoader(fake_dataset, batch_size=self.config["data"]["parts"][part]["batch_size"])
            fid_obj = piq.FID()
            real_features = fid_obj.compute_feats(self.dataloaders[part])
            fake_features = fid_obj.compute_feats(fake_dataloader)
            fid = fid_obj(fake_features, real_features)

            wandb.log({f"{part} generator loss": total_g_loss}, step=epoch * self.len_epoch)
            wandb.log({f"{part} discriminator(real) loss": total_real_loss}, step=epoch * self.len_epoch)
            wandb.log({f"{part} discriminator(fake) loss": total_fake_loss}, step=epoch * self.len_epoch)
            wandb.log({f"{part} discriminator loss": total_d_loss}, step=epoch * self.len_epoch)
            wandb.log({f"{part} ssim": total_ssim}, step=epoch * self.len_epoch)
            wandb.log({f"{part} fid": fid}, step=epoch * self.len_epoch)
            fake_images = fake_images*0.5 + 0.5
            for j in range(5):
                wandb.log({f"{part} image_{j}": wandb.Image(fake_images[j])}, step=epoch * self.len_epoch)


    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch)
            self.evaluation(epoch)


    
