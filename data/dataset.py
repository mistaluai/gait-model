import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from data.data_preprocessor import load_gait_sequences
from utils.visualize_samples import visualize_gait_samples, visualize_optical_flow_samples


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
            transforms.Normalize(mean=[0.5], std=[0.5])  # for grayscale
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

    def compute_optical_flow(self, frame1, frame2):
        f1 = frame1.squeeze(0).numpy()
        f2 = frame2.squeeze(0).numpy()

        f1 = (f1 * 255).astype(np.uint8)
        f2 = (f2 * 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(f1, f2, None,
                                            pyr_scale=0.5,
                                            levels=5,
                                            winsize=11,
                                            iterations=5,
                                            poly_n=5,
                                            poly_sigma=1.1,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        return flow_tensor

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = self.label_to_index[item['label']]
        seq_paths = item['sequence']

        # Load and transform all frames
        frames = [self.load_and_transform_image(p) for p in seq_paths]
        original_seq_len = len(frames)
        # Enforce target sequence length
        if len(frames) < self.target_length:
            frames = self.pad_sequence(frames)
        elif len(frames) > self.target_length:
            frames = self.downsample_sequence(frames)

        # Choose output type
        if self.return_one:
            accumulated_flow = torch.zeros(2, *frames[0].shape[1:])  # [2, H, W]
            for i in range(len(frames) - 1):
                flow = self.compute_optical_flow(frames[i], frames[i + 1])
                accumulated_flow += flow
            frames = accumulated_flow  # [2, H, W]
        else:
            frames = torch.stack(frames)  # Tensor: [T, 1, H, W]

        if self.return_metadata:
            meta = {
                'subject': item['subject'],
                'angle': item['angle'],
                'trial': item['trial'],
                'source': item['source']
            }
            if self.return_one:
                return frames, label, meta
            return frames, label, original_seq_len ,meta

        if self.return_one:
            return frames, label

        return frames, label, original_seq_len

if __name__ == "__main__":
    df = load_gait_sequences("./gei_maps/Multiclass6", load_images=False)

    dataset = GaitSequenceDataset(
        df,
        return_one=True,
        return_metadata=False,
    )

    loader = DataLoader(dataset, batch_size=2)
    seq, label = next(iter(loader))
    print(seq.shape)

    # visualize_gait_samples(dataset, n=5)
    visualize_optical_flow_samples(dataset, n=5)