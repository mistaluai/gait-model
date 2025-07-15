import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


def flow_to_rgb(flow):
    """
    Convert 2-channel flow to RGB image using HSV mapping.
    Args:
        flow: torch tensor of shape [2, H, W]
    Returns:
        RGB image as numpy array [H, W, 3] in uint8
    """
    flow_np = flow.cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2                # Hue: angle
    hsv[..., 1] = 255                    # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_optical_flow_samples(dataset, n=4):
    """
    Visualize n optical flow samples from a GaitSequenceDataset instance with return_one=True.

    Args:
        dataset (GaitSequenceDataset): the dataset object.
        n (int): number of samples to visualize.
    """
    assert dataset.return_one, "Dataset must be initialized with return_one=True to visualize flow."

    plt.figure(figsize=(n * 3, 3))

    for i in range(n):
        data = dataset[i]
        flow_tensor, label = data[:2]

        if isinstance(label, torch.Tensor):
            label = label.item()

        rgb_flow = flow_to_rgb(flow_tensor)  # Convert [2, H, W] → [H, W, 3]

        ax = plt.subplot(1, n, i + 1)
        ax.imshow(rgb_flow)
        ax.axis('off')
        ax.set_title(f"Class: {label}")

    plt.tight_layout()
    plt.show()

def visualize_gait_samples(dataset, n=4, return_one=None):
    """
    Visualize n samples from a GaitSequenceDataset instance.

    Args:
        dataset (GaitSequenceDataset): the dataset object.
        n (int): number of samples to visualize.
        return_one (bool): if None, use dataset.return_one;
                           otherwise, override it temporarily.
    """
    # Determine whether to show single or sequence
    use_single = dataset.return_one if return_one is None else return_one

    # Create a figure
    plt.figure(figsize=(n * 3, 3 if use_single else 4))

    for i in range(n):
        data = dataset[i]
        img_tensor, label = data[:2]

        if isinstance(label, torch.Tensor):
            label = label.item()

        if not use_single:
            # Sequence mode: show middle frame
            img_tensor = img_tensor[len(img_tensor) // 2]

        img = img_tensor.squeeze().numpy()  # shape: (H, W)

        ax = plt.subplot(1, n, i + 1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Class: {label}")

    plt.tight_layout()
    plt.show()