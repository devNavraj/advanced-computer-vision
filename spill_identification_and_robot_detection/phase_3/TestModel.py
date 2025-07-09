import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
# Once again I show my torch cuda device. Proof that my gaming GPU is not just for gaming
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))


# Here I just prep my data as image paths and respective labels
base_folder="LabPictures"
classes = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
num_classes=len(classes)
print(classes)
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

TestModel = models.resnet50(pretrained=False)
TestModel.fc = nn.Linear(TestModel.fc.in_features, 21) #<---if training the 3Category Model, this is commented out

state_dict = torch.load('21FineCategoryModel.pth', map_location=device) # we use '3FineCategoryModel.pth' when training 3Category Model

TestModel.load_state_dict(state_dict,strict=False)

TestModel.to(device)

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

# This transform is for the test and val data, where non of the augmentation should occur, but the images need to be cropped and resized still
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
# Create dataset objects
dataset = ImageDataset(image_paths, labels, transform=test_transform)

# Dataloaders
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
TestModel.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = TestModel(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    del images, labels, outputs, predicted
    torch.cuda.empty_cache()

accuracy = 100 * correct / total
print(f"Test Accuracy for 21 Category Model on Lab Images: {accuracy:.2f}%")