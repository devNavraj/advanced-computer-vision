# Import packages
import os
import pickle
import tarfile
import requests
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Avoid Jupyter Kernel error caused by multiple OpenMP runtimes being loaded into the process
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================
# Constants and Paths
# ============================
DATASET_DIR = 'fgvc_aircraft'
DATASET_URL = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
# Inside the extracted archive, files reside under this folder:
DATA_DIR = os.path.join('fgvc-aircraft-2013b', 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_PICKLE = os.path.join(DATA_DIR, 'labels.pkl')

VARIANTS_TXT = os.path.join(DATA_DIR, 'variants.txt')

# ============================
# Dataset Class Using Standardized Splits
# ============================
class FGVCAircraft(Dataset):
    """
    A PyTorch Dataset for FGVC-Aircraft that uses standardized splits: train, validation, test.

    The FGVC Aircraft dataset was originally introduced by Maji et al., (2013).

    Maji et al., (2013) recommend cropping the images using the bounding box information,
    to remove copyright information and ensure that only one plane is visible in the image.

    **References**
    1. Maji, S., Rahtu, E., Kannala, J., Blaschko, M., & Vedaldi, A. (2013). *Fine-grained visual classification of aircraft*. arXiv. https://arxiv.org/abs/1306.5151
    2. [http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
    
    The implementation:
      - Downloads and extracts the dataset (if needed).
      - Reads the three variant files (train, validation, test) and stores a list of tuples:
            (image_name, label, split)
      - In data loading, it filters the data based on the selected mode.
      - Optionally applies on-the-fly bounding box cropping.
      
    Args:
        root (str): Root directory to store the dataset.
        mode (str): One of 'train', 'validation', 'test', or 'all'.
                    (For 'validation', the internal split label is 'valid'.)
        transform: Transformations to apply on images.
        download (bool): If True, downloads the dataset if not present.
        bounding_box_crop (bool): If True, applies cropping based on bounding box info.
        remove_banner (bool): If True, removes banner containing copyright information
    """
    def __init__(self, root, mode='all', transform=None, download=False, bounding_box_crop=False, remove_banner=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.bounding_box_crop = bounding_box_crop
        self.remove_banner = remove_banner

        if self.bounding_box_crop and self.remove_banner:
            raise ValueError("Cannot enable both Bounding Box Crop and Banner Removal")
            
        # Full dataset path (e.g., "./data/fgvc_aircraft")
        self.dataset_path = os.path.join(self.root, DATASET_DIR)
        self.labels_pickle = os.path.join(self.dataset_path, LABELS_PICKLE)
        self.variants_txt = os.path.join(self.dataset_path, VARIANTS_TXT)
        
        if not self._check_exists() and download:
            self._download_dataset()
        else:
            print("================Dataset already exists.================")
        
        if not os.path.exists(self.labels_pickle):
            # Build the label pickle with standardized splits.
            print("Extracting labels from variant files...")
            self._build_label_pickle()
        else:
            print("===============Labels already extracted.===============")

        # Validate mode input
        allowed_modes = ['train', 'validation', 'test', 'trainval', 'all']
        assert mode in allowed_modes, f"mode should be one of: {allowed_modes}."
        # Internally, we use 'valid' instead of 'validation'
        self.mode = 'valid' if mode == 'validation' else mode
        self._load_data(self.mode)

    def _check_exists(self):
        images_path = os.path.join(self.dataset_path, IMAGES_DIR)
        return (os.path.exists(self.dataset_path) and 
                os.path.exists(images_path)) #and 
                # os.path.exists(self.labels_pickle))

    def _download_dataset(self):
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, DATASET_DIR), exist_ok=True)
        tar_path = os.path.join(self.root, DATASET_DIR, os.path.basename(DATASET_URL))
        if not os.path.exists(tar_path):
            print("Downloading FGVC-Aircraft dataset (this may take a while)...")
            req = requests.get(DATASET_URL, stream=True)
            with open(tar_path, 'wb') as f:
                for chunk in req.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=os.path.join(self.root, DATASET_DIR))
        os.remove(tar_path)
        print("Download and extraction complete.")

    def _build_label_pickle(self):
        """
        Reads the three variant files, and creates a list of (image_file_name, label, split) tuples.
        """
        # List of tuples: (variant file name, split label)
        variant_files = [
            ('images_variant_train.txt', 'train'),
            ('images_variant_val.txt', 'valid'),
            ('images_variant_test.txt', 'test'),
            ('images_variant_trainval.txt', 'trainval')
        ]
        images_labels = []
        data_dir = os.path.join(self.dataset_path, DATA_DIR)
        
        for fname, split in variant_files:
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        label = ' '.join(parts[1:])
                        images_labels.append((image_name, label, split))
                        
        # Save the image-label-split list as a pickle.
        with open(self.labels_pickle, 'wb') as f:
            pickle.dump(images_labels, f)

    def _load_data(self, mode='train'):
        """
        Loads image-label-split tuples from the pickle and filters them by the selected split.
        If mode is 'all', all variants are used.
        Optionally loads bounding box info for cropping.
        """
        data_dir = os.path.join(self.dataset_path, DATA_DIR)
        with open(self.labels_pickle, 'rb') as f:
            all_variants = pickle.load(f)
        # Filter based on mode.
        if mode == 'all':
            filtered = all_variants
        elif mode == 'trainval':
            # For 'trainval' split, we use training+validation sets
            filtered = [variant for variant in all_variants if variant[2] == 'trainval']
        else:
            # For 'train', 'valid' (validation) or 'test', use only that split.
            filtered = [variant for variant in all_variants if variant[2] == mode]
            
        # Load bounding box info if cropping is enabled.
        if self.bounding_box_crop:
            self.bounding_boxes = {}
            bbox_file = os.path.join(data_dir, 'images_box.txt')
            bbox_content = {}
            with open(bbox_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bbox_content[parts[0]] = (int(parts[1]), int(parts[2]),
                                                  int(parts[3]), int(parts[4]))
                        
        # Build the final data list as tuples: (image_path, label_index)
        self.data = []
        
        # Build a sorted list of variants from the filtered data.
        unique_variants = sorted(list({label for _, label, _ in filtered}))

        # Build a label map from the sorted list of variants (from the pickle)
        self.label_map = {label: idx for idx, label in enumerate(unique_variants)}

        # Create a reverse mapping from index to label text.
        self.rev_label_map = {idx: label for label, idx in self.label_map.items()}

        for image_name, label, _ in filtered:
            image_path = os.path.join(self.dataset_path, IMAGES_DIR, image_name + '.jpg')
            if self.bounding_box_crop:
                # Use image name as key (bbox file keys are image names)
                self.bounding_boxes[image_path] = bbox_content.get(image_name, None)
            label_idx = self.label_map[label]
            self.data.append((image_name, image_path, label_idx, label))

    def __getitem__(self, index):
        image_name, image_path, label, class_label = self.data[index]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Unable to open image {image_path}: {e}")
        
        # Crop to bounding box
        if self.bounding_box_crop:
            bbox = self.bounding_boxes.get(image_path, None)
            if bbox is not None:
                image = image.crop(bbox)
        # Remove copyright banner (last 20 pixel)
        if self.remove_banner and not self.bounding_box_crop:
            width, height = image.size
            image = image.crop((0, 0, width, height - 20))  # Remove bottom 20px banner
        
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

# ============================
# Visualization Function
# ============================
def visualize_samples(dataset, num_samples=6, title="Dataset Samples"):
    """
    Visualizes a few sample images from the dataset with denormalization.
    """
    # Randomly select distinct indices
    samples = random.sample(range(len(dataset)), num_samples)
    
    # Determine grid size:
    if num_samples <= 5:
        nrows = 1
        ncols = num_samples
    else:
        ncols = 5
        nrows = np.ceil(num_samples / ncols).astype(int)
    
    # Create a grid with nrows x 5 subplots.
    fig, axes = plt.subplots(nrows, 5, figsize=(5 * 4, nrows * 4))
    # Flatten axes to iterate over easily.
    if nrows * 5 > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, idx in enumerate(samples):
        image, label_idx = dataset[idx]

        # Extract the image filename from the image path stored in dataset.data.
        if hasattr(dataset, 'data'):
            img_name, _, _, variant_name = dataset.data[idx]
            # img_name = os.path.basename(img_path)

        # Convert tensor to NumPy array and reverse normalization.
        image_np = image.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        axes[i].imshow(image_np)
        axes[i].set_title(f"Label: {variant_name}\nLabel Index: {label_idx}\nImage Filename: {img_name}")
        axes[i].axis('off')

    # Remove any extra (unused) axes so no blank subplots are shown.
    for j in range(num_samples, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ====================================
# Main Execution - Dataset Observation
# ====================================
if __name__ == "__main__":
    # Define the transformation pipeline.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    # Create a training dataset (standardized split from images_variant_train.txt)
    train_dataset = FGVCAircraft(root='./data', mode='train', transform=transform,
                                 download=True, bounding_box_crop=True, remove_banner=False)
    print(f"Number of training set images: {len(train_dataset)}")
    visualize_samples(train_dataset,num_samples=10, title="Training Set Samples")

    # Create a test dataset (from images_variant_test.txt)
    test_dataset = FGVCAircraft(root='./data', mode='test', transform=transform,
                                download=False, bounding_box_crop=False, remove_banner=True)
    print(f"Number of testing set images: {len(test_dataset)}")
    visualize_samples(test_dataset, num_samples=5, title="Testing Set Samples")