import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from dataset_collection_preparation import FGVCAircraft


# Set reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset setup
training_dataset = FGVCAircraft(root='./data', mode='trainval', transform=transform,
                       download=False, bounding_box_crop=False, remove_banner=True)
print(f"Number of training set images: {len(training_dataset)}")

num_classes = len(training_dataset.label_map)
class_label = random.choice(range(num_classes))  # Random class label selection

_, _, _, class_name = training_dataset.data[class_label]
print(f"Selected class index: {class_label}, Name: {class_name}")

# Filter dataset to only images of the selected class
class_indices = [i for i, (_, label) in enumerate(training_dataset) if label == class_label]
filtered_dataset = Subset(training_dataset, class_indices)
print(f"Number of images in selected class subset: {len(filtered_dataset)}")

# DataLoader for GAN training on the selected  class
dataloader = DataLoader(filtered_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Hyperparameters for GAN
nz = 100       # Size of latent noise vector
ngf = 64       # Base feature maps in generator
ndf = 64       # Base feature maps in discriminator
nc = 3         # Number of image channels (3 for RGB)

# DCGAN Implementation

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent vector Z of shape (nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),  # output shape: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),  # output shape: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),  # output shape: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),  # output shape: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()       # output shape: (nc) x 64 x 64, with pixels in [-1,1]
        )
    
    def forward(self, z):
        return self.main(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 64 x 64 image
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # output: ndf x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # output: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # output: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # output: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Output is 1x1 (single value). We'll apply Sigmoid in the training loop.
        )
    
    def forward(self, x):
        return self.main(x)
    
# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)

# Weight initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

netG.apply(weights_init)
netD.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCEWithLogitsLoss()  # using logits for stability
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))


# Training loop 
real_label = 1.0
fake_label = 0.0

for epoch in range(NUM_EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
        b_size = imgs.size(0)
        # Move data to device (CPU or GPU)
        real_imgs = imgs.to(device)  # on CPU for this example; use imgs.cuda() if GPU available
        
        ### Train Discriminator ###
        netD.zero_grad()
        # Real images
        labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_imgs).view(-1)              # discriminator predictions on real
        lossD_real = criterion(output, labels)    # real images labeled as 1
        lossD_real.backward()
        
        # Fake images
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_imgs = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake_imgs.detach()).view(-1)  # detach to avoid grad in G
        lossD_fake = criterion(output, labels)      # fake images labeled as 0
        lossD_fake.backward()
        optimizerD.step()  # update Discriminator
        
        # Total discriminator loss (optional logging)
        lossD = lossD_real + lossD_fake
        
        ### Train Generator ###
        netG.zero_grad()
        labels.fill_(real_label)                  # generator wants D to think fakes are real
        output = netD(fake_imgs).view(-1)         # recompute D's prediction on fakes (not detached this time)
        lossG = criterion(output, labels)         # want output to be 1 for fakes
        lossG.backward()
        optimizerG.step()
        
        # (Optional) print training stats occasionally
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i}/{len(dataloader)}] "
                  f"Loss_D: {lossD.item():.4f}, Loss_G: {lossG.item():.4f}")

# Switch generator to evaluation mode for inference
netG.eval()

# Generate 10 synthetic images
NUM_SAMPLES = 10
fixed_noise = torch.randn(NUM_SAMPLES, nz, 1, 1, device=device)  # fixed noise for reproducibility (or could be random each time)
fake_images = netG(fixed_noise)

# Denormalize from [-1,1] to [0,1] for viewing
fake_images_denorm = fake_images * 0.5 + 0.5

# Save and display the generated images
out_dir = "./generated_images"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(20, 8))
for j in range(NUM_SAMPLES):
    img = fake_images_denorm[j]
    vutils.save_image(img, f"{out_dir}/fake_{j}.png")  # save image
    # Plot the image
    np_img = img.cpu().detach().numpy().transpose(1, 2, 0)
    plt.subplot(2, 5, j+1)
    plt.imshow(np_img)
    plt.axis("off")
plt.tight_layout()
plt.show()
