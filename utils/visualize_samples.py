import matplotlib.pyplot as plt
import torch

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