from pathlib import Path
from PIL import Image
import numpy as np


class ImageFolderDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from directory, organizing samples and labels."""
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = class_idx
                self.idx_to_class[class_idx] = class_dir.name
                for img_path in class_dir.glob('*.jpg'):  # Adjust glob pattern for other image types
                    self.samples.append((img_path, class_idx))

        self.num_classes = len(self.class_to_idx)

    def _load_image(self, path):
        """Load an image from its path and apply optional transformations."""
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return np.array(img)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = self._load_image(img_path)
        label = np.zeros(self.num_classes)
        label[class_idx] = 1  # One-hot encoding
        return image, label

    def __len__(self):
        return len(self.samples)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        # Shuffle indices if shuffle is True
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        # Generator function to yield batches
        for start_idx in range(0, len(self.dataset), self.batch_size):
            if start_idx + self.batch_size > len(self.dataset) and start_idx < len(self.dataset):
                # Ensure we don't go out of bounds and grab remaining items for last batch
                end_idx = len(self.dataset)
            else:
                end_idx = start_idx + self.batch_size
            batch_indices = self.indices[start_idx:end_idx]
            batch = [self.dataset[idx] for idx in batch_indices]
            images, labels = zip(*batch)  # Unpack list of tuples
            # Stack images and labels to get batch arrays
            images_stack = np.stack(images, axis=0)
            labels_stack = np.stack(labels, axis=0)
            return images_stack, labels_stack
        raise StopIteration

    def __len__(self):
        # Returns the number of batches per epoch
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
