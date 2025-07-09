import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_collection_preparation import FGVCAircraft

# ------------------------------
# Pre-Trained ResNet-50 Classifier
# ------------------------------
class ResNet50Classifier(nn.Module):
    """
    A custom classifier based on pretrained ResNet-50.
    Replaces the final FC layer to match `num_classes`, with optional
    freezing of backbone weights for transfer learning.
    """
    def __init__(self, num_classes, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of target classes.
            freeze_backbone (bool): If True, do not update backbone weights.
        """
        super(ResNet50Classifier, self).__init__()
        # Load a ResNet-50 model pretrained on ImageNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze all backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Retrieve the input feature size of the original FC layer
        in_features = self.backbone.fc.in_features
        
        # Replace the original classification head with a custom MLP:
        # Dropout -> Linear(→768) -> ReLU -> Dropout -> Linear(→num_classes)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the backbone and new head.
        
        Args:
            x (Tensor): input batch tensor of shape [B, 3, H, W].
        Returns:
            Tensor: raw class logits of shape [B, num_classes].
        """
        return self.backbone(x)


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


# ------------------------------
# Training and Evaluating Pre-Trained ResNet-50 Classifier 
# ------------------------------
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
    
    # Image preprocessing pipelines

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    
    # Load FGVC-Aircraft datasets (train/val/test)
    print("Loading datasets...")
    train_dataset = FGVCAircraft(
        root='./data',
        mode='train',
        transform=transform,
        download=True,               # download if not present
        bounding_box_crop=False,
        remove_banner=True
    )
    valid_dataset = FGVCAircraft(
        root='./data',
        mode='validation',
        transform=transform,
        download=False,
        bounding_box_crop=False,
        remove_banner=True
    )
    test_dataset = FGVCAircraft(
        root='./data',
        mode='test',
        transform=transform,
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
    
    # ------------------------------
    # Fine-tune ResNet-50 (all layers trainable)
    # ------------------------------
    print("\nFine-tuning ResNet-50 (all layers trainable)...")
    resnet_model = ResNet50Classifier(num_classes=NUM_CLASSES, freeze_backbone=False)
    
    criterion = nn.CrossEntropyLoss()                         # multi-class classification loss
    optimizer = optim.Adam(resnet_model.parameters(), 
                           lr=1e-4, 
                           weight_decay=1e-4)                # Adam with L2 regularization
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    # Train and validate the model
    fine_tuned_model, history_ft = train_model(resnet_model, train_loader, valid_loader, 
                                               criterion, optimizer, num_epochs=NUM_EPOCHS, 
                                               model_name="resnet50_fine_tuned")
    
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
    
    # Plot ResNet-50 Model training curves
    plot_metrics(history_ft, "Fine Tuned ResNet-50 Pre-trained Model")
    
    # Evaluate on testing set
    test_loss_ft, test_accuracy_ft = evaluate_model(fine_tuned_model, test_loader, criterion, device)
    print(f"Fine-Tuned ResNet-50 Model - Test Loss: {test_loss_ft:.4f} | Test Accuracy: {test_accuracy_ft:.2f}%")

    # --------------------------------------------
    # Transfer Learning Setting (Freeze backbone)
    # --------------------------------------------
    print("\nTransfer Learning ResNet-50 (Freeze backbone)")
    transfer_model = ResNet50Classifier(num_classes=NUM_CLASSES, freeze_backbone=True)

    # Only parameters of the classifier will be updated.
    optimizer_transfer = optim.Adam(transfer_model.backbone.fc.parameters(), lr=0.0001, weight_decay=1e-4)

    start_time = time.time()
    transfer_model, history_tl = train_model(transfer_model, train_loader, valid_loader, 
                                            criterion, optimizer_transfer, num_epochs=NUM_EPOCHS, 
                                            model_name="resnet50_transfer_learning")
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")

    # Plot Transfer Learning Model training curves
    plot_metrics(history_tl, "ResNet50_Transfer_Learning")

    # Evaluate on testing set
    test_loss_tl, test_accuracy_tl = evaluate_model(transfer_model, test_loader, criterion, device)
    print(f"Transfer Learning Test Loss: {test_loss_tl:.4f} | Test Accuracy: {test_accuracy_tl:.2f}%")