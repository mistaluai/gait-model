import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from data.data_preprocessor import load_gait_sequences
from utils.visualize_samples import visualize_gait_samples


class GaitSequenceDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_to_index=None,
                 return_metadata=False, return_one=False, image_size=(224, 224)):
        """
        Args:
            dataframe (pd.DataFrame): Output from load_gait_sequences(...)
            transform (callable, optional): Transform to apply to each frame
            label_to_index (dict, optional): Map class labels to integers
            return_metadata (bool): Return subject/angle/etc.
            return_one (bool): Return only the middle frame
            image_size (tuple): Target size for padding and resizing (H, W)
        """
        self.data = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.return_metadata = return_metadata
        self.return_one = return_one
        self.image_size = image_size

        # Compute median sequence length
        self.target_length = int(np.median(self.data['sequence'].apply(len)))

        # Label to index mapping
        if label_to_index is None:
            classes = sorted(self.data['label'].unique())
            self.label_to_index = {cls: i for i, cls in enumerate(classes)}
        else:
            self.label_to_index = label_to_index

    def __len__(self):
        return len(self.data)

    def load_and_transform_image(self, path):
        img = Image.open(path).convert("L")  # Convert to grayscale
        img = self.transform(img)
        return img

    def pad_sequence(self, frames):
        pad_count = self.target_length - len(frames)
        if pad_count <= 0:
            return frames

        # Create black padding frame (grayscale)
        black = Image.new("L", self.image_size, 0)
        if self.transform:
            black = self.transform(black)
        else:
            black = transforms.ToTensor()(black)

        return [black] * pad_count + frames

    def downsample_sequence(self, frames):
        if len(frames) <= self.target_length:
            return frames

        indices = np.linspace(0, len(frames) - 1, self.target_length).astype(int)
        return [frames[i] for i in indices]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = self.label_to_index[item['label']]
        seq_paths = item['sequence']

        # Load and transform all frames
        frames = [self.load_and_transform_image(p) if isinstance(p, str) else p for p in seq_paths]
        original_seq_len = len(frames)
        # Enforce target sequence length
        if len(frames) < self.target_length:
            frames = self.pad_sequence(frames)
        elif len(frames) > self.target_length:
            frames = self.downsample_sequence(frames)

        # Choose output type
        if self.return_one:
            mid_idx = len(frames) // 2
            frames = frames[mid_idx]  # Tensor: [1, H, W]
        else:
            frames = torch.stack(frames)  # Tensor: [T, 1, H, W]

        if self.return_metadata:
            meta = {
                'subject': item['subject'],
                'angle': item['angle'],
                'trial': item['trial'],
                'source': item['source']
            }
            return frames, label, meta

        return frames, label, original_seq_len

if __name__ == "__main__":
    df = load_gait_sequences("./gei_maps/Multiclass6", load_images=False)

    transform = transforms.ToTensor()
    dataset = GaitSequenceDataset(
        df,
        transform=transform,
        return_one=False,
        return_metadata=False,
    )

    loader = DataLoader(dataset, batch_size=2)
    seq, label = next(iter(loader))
    print(seq.shape)

    visualize_gait_samples(dataset, n=5)