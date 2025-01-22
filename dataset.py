
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from device_detection import get_available_device

device = get_available_device()

class MultimodalSyntheticDataset(Dataset):
    def __init__(self, num_samples=500, num_features=10, image_size=(64, 64), num_classes=2):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.image_size = image_size
        self.num_classes = num_classes

        if num_features <= 1: 
            raise ValueError("Number of features must be >1 ")
        
        if num_samples <= 1: 
            raise ValueError("Number of samples must be >1 ")
        
        if num_samples % num_classes != 0: 
            raise ValueError("Number of samples must be divisible by the number of classes")
        
        if num_features % 2 != 0: 
            raise ValueError("The number of features must be an even number")

        samples_per_class = num_samples // num_classes

        # Generate synthetic structured data within specific ranges for each feature and label
        self.structured_data = np.zeros((num_samples, num_features), dtype=np.float32) 
        self.labels =  np.zeros((num_samples), dtype=np.int64) 
        
        # Creating specific random intervals for each features in classes
        for _class in range(num_classes):       
            print(f'Generating features for label {_class}')
            ranges_classes = [((_class * num_features + feature), (_class * num_features + feature)+0.95) for feature in range(num_features//2)]             
            print(f"Ranges {ranges_classes} ")
            for i in range(samples_per_class):
                sample_index = _class * samples_per_class + i
                self.labels[sample_index] = _class
                
                for feat in range(num_features): 
                    if feat < (num_features // 2):
                        range_start, range_finish = ranges_classes[feat]
                        self.structured_data[sample_index, feat] = round(np.random.uniform(range_start, range_finish), 2)
                    else:
                        self.structured_data[sample_index, feat] = round(np.random.uniform(100, 100000),2)

            # Generate synthetic image data
            self.image_data = self._generate_images()

    def _generate_images(self):
        """
        Generate synthetic images with simple patterns and color based on the label.
        """

        #Calculating proportion based on the number of unique labels
        unique_labels = np.unique(self.labels).size
        
        # Normalizing unique labels colors between  1 and 240 
        norm_uniq_labels_colors = [int(np.interp(i, (0, unique_labels), (1, 240))) for i in range(unique_labels)]
        
        # Normalizing unique labels circle size between  25 and 54 
        norm_uniq_labels_circles = [int(np.interp(i, (0, unique_labels), (self.image_size[1]//10, self.image_size[1]//2))) for i in range(unique_labels)]

        images = []
        for label in self.labels:
            img = Image.new('RGB', self.image_size, (255, 255, 255))  # White background
            draw = ImageDraw.Draw(img)
            
            draw.ellipse((10, 10, norm_uniq_labels_circles[label], norm_uniq_labels_circles[label]), fill=(norm_uniq_labels_colors[label], norm_uniq_labels_colors[label], norm_uniq_labels_colors[label]))  # Green circle
            
            images.append(ToTensor()(img).numpy())

        return np.array(images, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.structured_data[idx], device=device),
            torch.tensor(self.image_data[idx], device=device),
            torch.tensor(self.labels[idx], device=device),
        )


# Example usage
# num_samples = 1000
# num_features = 10
# image_size = (64, 64)
# num_classes = 2

# dataset = MultimodalSyntheticDataset(num_samples=num_samples, num_features=num_features, image_size=image_size, num_classes=num_classes)

