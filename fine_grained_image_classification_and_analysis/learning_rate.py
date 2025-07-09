import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_collection_preparation import FGVCAircraft

# -------------------------------------------------
# SkipBlock: Two 3x3 Convs + Skip Connection
# -------------------------------------------------
class SkipBlock(nn.Module):
    """
    A skip-based block for deep CNNs, using two 3x3 convs plus a shortcut.
    stride=2 if we need to downsample spatial size, otherwise stride=1.
    """
    expansion = 1  # If you had a bottleneck style, you'd set expansion=4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the in/out dims (or stride) differ, we adjust the shortcut
        self.downsample = downsample

    def forward(self, x):
        # Save the input for the skip path
        identity = x
        if self.downsample is not None:
            # Adjust identity if shape changes
            identity = self.downsample(x)

        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip
        out += identity
        out = F.relu(out, inplace=True)
        return out


# -------------------------------------------------
# Self-designed CNN model for Fine-Grained Images
# -------------------------------------------------
class FineGrainedCNN(nn.Module):
    """
    A deep CNN for fine-grained tasks. 
    Uses skip connections to help training at greater depth.
    """
    def __init__(self, num_classes=102):
        super(FineGrainedCNN, self).__init__()
        self.in_channels = 64

        # Initial layers: 7x7 conv, stride=2, BN, ReLU, then max pool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # ~ /4 resolution

        # Stage 1: 64 -> 64
        self.layer1 = self._make_layer(SkipBlock, out_channels=64,  blocks=2, stride=1)
        # Stage 2: 64 -> 128
        self.layer2 = self._make_layer(SkipBlock, out_channels=128, blocks=2, stride=2)
        # Stage 3: 128 -> 256
        self.layer3 = self._make_layer(SkipBlock, out_channels=256, blocks=2, stride=2)
        # Stage 4: 256 -> 512
        self.layer4 = self._make_layer(SkipBlock, out_channels=512, blocks=2, stride=2)

        # Global average pool + linear
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SkipBlock.expansion, num_classes)

        # Initialize weights
        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates one 'stage' with 'blocks' skip-blocks. 
        The first block may have stride=2 if we need to downsample.
        """
        downsample = None
        # If we are changing channel dims or using stride=2, define a 1x1 conv for the skip
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Additional blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial stage
        x = self.conv1(x)   # B, 64, H/2, W/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)    # B, 64, H/4, W/4

        # Four skip-based stages
        x = self.layer1(x)  # out channels = 64
        x = self.layer2(x)  # out channels = 128
        x = self.layer3(x)  # out channels = 256
        x = self.layer4(x)  # out channels = 512

        # Classification
        x = self.avgpool(x)           # => [B, 512, 1, 1]
        x = torch.flatten(x, 1)       # => [B, 512]
        x = self.fc(x)                # => [B, num_classes]
        return x

# ------------------------------
# Implement Improved LR
# ------------------------------
def get_advanced_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    # Phase 1: Linear Warmup
    warmup_steps = int(total_steps * warmup_ratio)
    warmup = LinearLR(
        optimizer, 
        start_factor=1e-5,  # Start from 1e-5
        end_factor=1.0,     # Reach initial_lr
        total_iters=warmup_steps
    )

    # Phase 2: OneCycle Policy
    onecycle = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps - warmup_steps,
        pct_start=0.3,
        final_div_factor=1e4
    )

    # Combine phases
    return SequentialLR(
        optimizer,
        schedulers=[warmup, onecycle],
        milestones=[warmup_steps]
    )

# ------------------------------
# Training & Evaluation Functions
# ------------------------------
def train_model(model, train_loader, valid_loader, criterion, optimizer,
                num_epochs=20, model_name="model"):
    """
    Train the model and track metrics over epochs.
    Saves the best model based on validation accuracy.

    Args:
        model (nn.Module): the neural network to train.
        train_loader (DataLoader): provides training batches.
        valid_loader (DataLoader): provides validation batches.
        criterion: loss function (e.g., CrossEntropyLoss).
        optimizer: optimizer (e.g., Adam).
        num_epochs (int): number of training epochs.
        model_name (str): name prefix for logging.

    Returns:
        best_model (nn.Module): model with highest validation accuracy.
        history (dict): training/validation losses & accuracies per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Lists to store per-epoch metrics
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    best_valid_accuracy = 0.0
    best_model_wts = None
    
    # Total steps = epochs * batches_per_epoch
    total_steps = num_epochs * len(train_loader)  
    scheduler = get_advanced_scheduler(optimizer, total_steps)

    # Epoch loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training phase ---
        model.train()   # enable dropout, batchnorm updates
        epoch_train_loss, correct, total = 0.0, 0, 0
        
        # Iterate over training batches with progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)                  # forward
            loss = criterion(outputs, labels)        # compute loss
            loss.backward()                          # backpropagate
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()                         # update weights
            scheduler.step()                         # call scheduler
            
            # Accumulate loss and accuracy
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Compute average training metrics
        train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # --- Validation phase ---
        valid_loss, valid_accuracy = evaluate_model(model, valid_loader, criterion, device)
        
        # Save best model if validation accuracy improved
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_wts = model.state_dict()
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {valid_loss:.4f} | Validation Accuracy: {valid_accuracy:.2f}%")
        print("-" * 50)
    
    # Load best weights before returning
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'valid_loss': valid_losses,
        'valid_accuracy': valid_accuracies
    }
    
    print(f"Best Validation Accuracy of {model_name}: {best_valid_accuracy:.2f}%")
    return model, history


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset (no weight updates).

    Args:
        model (nn.Module): the neural network in eval mode.
        data_loader (DataLoader): provides batches to evaluate.
        criterion: loss function.
        device: 'cpu' or 'cuda'.

    Returns:
        avg_loss (float), accuracy_pct (float)
    """
    model.eval()  # disable dropout, fix batchnorm
    loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return loss / len(data_loader), 100 * correct / total


def plot_metrics(history, model_name):
    """
    Plot and save training vs. validation loss and accuracy curves.

    Args:
        history (dict): contains 'train_loss', 'valid_loss',
                        'train_accuracy', 'valid_accuracy'.
        model_name (str): used for plot titles and saved filename.
    """
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["valid_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# Training Self-Designed CNN Model with Improved Learning Rate
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    NUM_EPOCHS = 30        # total training epochs
    BATCH_SIZE = 32        # samples per gradient update
    
    # Image transformation pipeline with strong data augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load FGVC-Aircraft datasets (train/val/test)
    print("Loading datasets...")
    train_dataset = FGVCAircraft(
        root='./data',
        mode='train',
        transform=train_transform,
        download=True,               # download if not present
        bounding_box_crop=False,
        remove_banner=True
    )
    valid_dataset = FGVCAircraft(
        root='./data',
        mode='validation',
        transform=test_transform,
        download=False,
        bounding_box_crop=False,
        remove_banner=True
    )
    test_dataset = FGVCAircraft(
        root='./data',
        mode='test',
        transform=test_transform,
        download=False,
        bounding_box_crop=False,
        remove_banner=True
    )
    print("###############Datasets loaded.###############")
    
    # Create DataLoader objects for batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Determine number of target classes from dataset
    NUM_CLASSES = len(train_dataset.label_map)
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Print dataset sizes
    print(f"Train set size:      {len(train_dataset)} samples")
    print(f"Validation set size: {len(valid_dataset)} samples")
    print(f"Test set size:       {len(test_dataset)} samples")
    
    # Initialize model, loss function and optimizer
    print("\nInitializing model...")
    custom_cnn = FineGrainedCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    
    # Use a smaller learning rate and weight decay for regularization
    optimizer = optim.Adam(custom_cnn.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Print model summary
    print(f"\nModel Architecture: {custom_cnn}")
    print(f"Total parameters: {sum(p.numel() for p in custom_cnn.parameters())}")
    
    # Train the model
    print(f"\nTraining model for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    model, history = train_model(
        custom_cnn, 
        train_loader, 
        valid_loader, 
        criterion, 
        optimizer, 
        num_epochs=NUM_EPOCHS, 
        model_name="custom_cnn"
    )
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
    
    # Plot self-designed CNN Model training curves
    plot_metrics(history, "Self Designed CNN Model With Improved LR")
    
    # Evaluate on testing set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")