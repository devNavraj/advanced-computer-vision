
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
from PIL import Image
import matplotlib.pyplot as plt
"""
This code is similar to the model maker, but instead of tuning a new Resnet50,
It loads the 21CategoryModel state_dict into it, and finetunes 
with the Lab Pictures, into the 3Category Model.
"""



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))
base_folder="LabPictures"
classes = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
num_classes=len(classes)

#I just load image paths and labels from the the root directory
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

image_paths, labels = [], []
for cls in classes:
    class_folder = os.path.join(base_folder, cls)
    for img_name in os.listdir(class_folder):
        if img_name.endswith(".jpg"):
            image_paths.append(os.path.join(class_folder, img_name))
            labels.append(class_to_idx[cls])

OriginalModel = models.resnet50(pretrained=False)

# This is the tricky part. First, I load the state_dict into an object
state_dict = torch.load('21CategoryModel.pth', map_location=device)

# Then I remove final layer weights of the resnet50 model (since we are switching from 21 Categories to 3 Categories)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)

# Here I load the state_dict into the model
OriginalModel.load_state_dict(state_dict, strict=False)

# And Replace classifier head for 3-class output
OriginalModel.fc = nn.Linear(OriginalModel.fc.in_features, 3)

OriginalModel.to(device)

# Here I open all the parameters for fine-tuning.
for param in OriginalModel.parameters():
    param.requires_grad = True

""" 
Just like in Assignment 1,
I'll be doing a per-layer learning rate, giving higher learning rates to my later layers, and lower learning rates to my
earlier layers. 
"""
r50layerwise_lrs = {
    "fc": 1e-3,        # Highest learning rate for final layer
    "layer4": 1e-4,    # Moderate learning rate for the last residual block
    "layer3": 5e-5,    # Lower learning rate for mid-level features
    "layer2": 1e-5,    # Even lower learning rate for early mid-level features
    "layer1": 1e-6,    # Very low learning rate for shallow layers (edge/textures)
    "conv1": 1e-7      # Barely moving (keeps basic low-level features intact)
}

# Here I prepare parameter groups for optimizer
r50param_groups = []
for name, param in OriginalModel.named_parameters():
    for layer, lr in r50layerwise_lrs.items():
        if layer in name:
            r50param_groups.append({"params": param, "lr": lr})
            break  # Once assigned to a group, move to the next parameter

# This is my own custom Dataset object, its not too complex since image paths and labels have already been extracted.
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
#transform for the test data
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Resize(256),                      
    transforms.CenterCrop(224), 
    transforms.ToTensor(),              
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# This transform is for the test and val data, where non of the augmentation should occur, but the images need to be cropped and resized still
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# This just ensures that all data is well represented across all classes and across all 3 sets
def stratified_split(image_paths, labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    random.seed(seed)
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        total = len(indices)
        t_end = int(train_ratio * total)
        v_end = t_end + int(val_ratio * total)
        train_idx.extend(indices[:t_end])
        val_idx.extend(indices[t_end:v_end])
        test_idx.extend(indices[v_end:])

    # Use indices to gather data
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

# Split them
(train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = stratified_split(image_paths, labels)

# Create dataset objects
train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
val_dataset = ImageDataset(val_paths, val_labels, transform=test_transform)  # basic transform if needed
test_dataset = ImageDataset(test_paths, test_labels, transform=test_transform)  # same here

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# I define my loss and learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(r50param_groups)

#Now I train my model's final layer. Originally set at 1000 and rely on early stopping,
num_epochs = 1000

#Here I setup some parameters for early stopping
best_val_loss = float('inf')  # Initialize with a very large value
patience_counter = 0  # Counter for patience
patience = 0  # Patience of 0 epoch
track_loss=[] #for graphing my Loss


for epoch in range(num_epochs):
    OriginalModel.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs =OriginalModel(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    OriginalModel.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0 

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = OriginalModel(images)
            loss = criterion(outputs, labels)  # Compute validation loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate the average validation loss
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    track_loss.append(avg_val_loss)
    # Check for improvement in validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss  # Update the best validation loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1  # Increment the counter if there's no improvement

    # Early stopping check
    if patience_counter > patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break  # Stop training

    # I keep getting Out of Memory issues so I need to clear somethings every epoch.
    del images, labels, outputs, loss, predicted
    torch.cuda.empty_cache()
#Testing Phase for Transfer Learning ResNet Model
plt.plot(range(len(track_loss)),track_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss for Our Model (3 Categories Fine Tuned)")
plt.savefig("3CategoriesLoss.png")

torch.save(OriginalModel.state_dict(), "3FineCategoryModel.pth")
print("Model Saved!")

OriginalModel.eval()  # Set model to evaluation mode for testing on test dataset
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = OriginalModel(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    del images, labels, outputs, predicted
    torch.cuda.empty_cache()

accuracy = 100 * correct / total
print(f"Test Accuracy for Transfer Learning: {accuracy:.2f}%")