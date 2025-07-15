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
    hsv[..., 0] = ang / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_optical_flow_samples(dataset, n=2, max_flows_per_sample=3):
    """
    Visualize optical flow maps from temporal flow sequences.

    Args:
        dataset: GaitOpticalFlowDataset with temporal flow [T-1, 2, H, W]
        n (int): number of samples to visualize
        max_flows_per_sample (int): max number of time steps (flows) per sample
    """
    sample = dataset[0][0]
    T = sample.shape[0]  # T-1 flows per sample
    max_t = min(T, max_flows_per_sample)

    fig, axes = plt.subplots(nrows=n, ncols=max_t, figsize=(max_t * 3, n * 3))

    if n == 1: axes = [axes]
    if max_t == 1: axes = [[ax] for ax in axes]

    for i in range(n):
        flow_seq, label = dataset[i][:2]
        for t in range(max_t):
            rgb = flow_to_rgb(flow_seq[t])  # [2, H, W] → [H, W, 3]
            ax = axes[i][t]
            ax.imshow(rgb)
            ax.axis('off')
            ax.set_title(f"t={t}→{t+1}")

        # Label the left-most image with the class
        axes[i][0].set_ylabel(f"Class: {label}", fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_gait_sequence_samples(dataset, n=4):
    """
    Visualize n grayscale gait sequence samples.
    Supports both full sequences and single frame mode.

    Args:
        dataset (GaitFrameSequenceDataset): the dataset object.
        n (int): number of samples to visualize.
    """
    is_single_frame = isinstance(dataset[0][0], torch.Tensor) and dataset[0][0].dim() == 3

    plt.figure(figsize=(n * 3, 3 if is_single_frame else 4))

    for i in range(n):
        data = dataset[i]
        img_tensor, label = data[:2]

        if isinstance(label, torch.Tensor):
            label = label.item()

        if not is_single_frame:
            # Sequence mode: show middle frame
            img_tensor = img_tensor[len(img_tensor) // 2]

        img = img_tensor.squeeze().numpy()  # shape: (H, W)

        ax = plt.subplot(1, n, i + 1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Class: {label}")

    plt.tight_layout()
    plt.show()