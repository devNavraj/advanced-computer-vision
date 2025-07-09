import torch
import torch.nn.functional as F
from torch import optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from dataset_collection_preparation import FGVCAircraft

# =========================================
# DeepDream with ResNet-50
# Algorithm: Multi-octave DeepDream
# =========================================

def deprocess(img_t):
    """
    Convert a tensor back to a NumPy image for visualization.

    Args:
        img_t: Tensor of shape [1, 3, H, W], normalized to ImageNet stats.

    Returns:
        NumPy array of shape [H, W, 3] with values in [0, 1].
    """
    # Undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = img_t.cpu().detach().squeeze(0)  # Remove batch dim
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    # Convert from [C,H,W] to [H,W,C]
    return img.permute(1, 2, 0).numpy()


class DeepDream:
    """
    DeepDream engine using a feature extractor (ResNet-50 conv layers).

    Implements multi‑octave DeepDream gradient ascent on a feature extractor.
    """

    def __init__(self, model, device):
        """
        Initialize with a pretrained feature model.

        Args:
            model: Feature-extractor module (e.g., ResNet conv layers).
            device: Torch device for computation.
        """
        self.model = model.to(device).eval()  # Set to evaluation mode
        self.device = device

    def _forward_to(self, x, layer_name):
        """
        Forward-pass up to a named module, return its activation.

        Args:
            x: Input tensor [B, C, H, W].
            layer_name: Name of the module in model._modules.

        Returns:
            Activation tensor from the specified layer.
        """
        for name, module in self.model._modules.items():
            x = module(x)
            if name == layer_name:
                return x
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    def dream(
        self,
        base_img,
        layer,
        iterations=20,
        lr=0.01,
        octave_scale=1.4,
        num_octaves=3,
        jitter=32
    ):
        """
        Perform multi-octave gradient ascent (DeepDream).

        Args:
            base_img: Preprocessed image tensor [1,3,H,W].
            layer: Target layer name to amplify.
            iterations: Steps per octave for ascent.
            lr: Learning rate for gradient updates.
            octave_scale: Scale factor between octaves.
            num_octaves: Number of scales to process.
            jitter: Max pixel shift each step for stability.

        Returns:
            Dreamed image tensor [1,3,H,W] on device.
        """
        # Build a pyramid of scaled images (smallest → original)
        octaves = [base_img]
        for _ in range(num_octaves - 1):
            small = F.interpolate(
                octaves[-1],
                scale_factor=1/octave_scale,
                mode='bilinear',
                align_corners=True
            )
            octaves.append(small)
        octaves = octaves[::-1]  # reverse to smallest→largest

        detail = torch.zeros_like(octaves[0], device=self.device)

        # Process each octave
        for octave in octaves:
            # Upsample detail to current octave size
            if detail.shape[-2:] != octave.shape[-2:]:
                detail = F.interpolate(
                    detail,
                    size=octave.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

            # Combine base and detail, enable grads
            img = (octave + detail).detach().requires_grad_(True)
            optimizer = optim.Adam([img], lr=lr)

            # Gradient ascent loop
            for _ in range(iterations):
                optimizer.zero_grad()

                # Apply random shift (jitter)
                ox, oy = np.random.randint(-jitter, jitter+1, 2)
                img.data = torch.roll(img.data, shifts=(ox, oy), dims=(-1, -2))

                # Forward to target layer
                activation = self._forward_to(img, layer)
                # Loss: negative L2 norm (we want to maximize norm)
                loss = -activation.norm()
                loss.backward()

                # Normalize gradient and step
                grad = img.grad.data
                img.data += lr * grad / (grad.std() + 1e-8)

                # Undo the jitter shift
                img.data = torch.roll(img.data, shifts=(-ox, -oy), dims=(-1, -2))

            # Extract detail for next octave
            detail = (img.detach() - octave)

        # Return the final dreamed image
        return (octaves[-1] + detail).detach()
    

if __name__ == "__main__":
    """
    Main pipeline:
      1. Load ResNet-50 convolutional layers.
      2. Read & preprocess an input image.
      3. Dream on specified layers.
      4. Visualize results side by side.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained ResNet-50 and extract conv layers
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    features = torch.nn.Sequential(OrderedDict([
        ('conv1', resnet.conv1),
        ('bn1', resnet.bn1),
        ('relu', resnet.relu),
        ('maxpool', resnet.maxpool),
        ('layer1', resnet.layer1),
        ('layer2', resnet.layer2),
        ('layer3', resnet.layer3),
        ('layer4', resnet.layer4)
    ]))
    dreamer = DeepDream(features, device)

    # This transform resizes, crops, converts to tensor, and normalizes for ResNet.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    
    # Instantiate the dataset (e.g. train split).  
    dataset = FGVCAircraft(
        root='./data',
        mode='train',
        transform=transform,
        download=False,
        bounding_box_crop=False,
        remove_banner=True
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    
    # ------------------------------------------------------------------------------
    # Fetch one sample & apply DeepDream
    # ------------------------------------------------------------------------------
    # Get a single preprocessed image tensor [1,3,224,224]
    base_img, label_idx = next(iter(loader))
    base_img = base_img.to(device)
    
    # Print which aircraft variant we're dreaming on
    class_variant = dataset.rev_label_map[int(label_idx)]
    print(f"DeepDream on variant: {class_variant}")
    
    # Choose layers to dream on
    target_layers = ['layer1', 'layer2', 'layer3']
    dreamed_outputs = []
    
    for layer in target_layers:
        print(f"  → dreaming on {layer} …")
        dreamed = dreamer.dream(
            base_img=base_img,
            layer=layer,
            iterations=30,
            lr=0.02,
            octave_scale=1.4,
            num_octaves=4,
            jitter=16
        )
        dreamed_outputs.append(deprocess(dreamed))
    
    # Plot results side‑by‑side
    fig, axes = plt.subplots(1, len(target_layers), figsize=(15,5))
    for ax, img, layer in zip(axes, dreamed_outputs, target_layers):
        ax.imshow(img)
        ax.set_title(f"DeepDream — {layer}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()