import torch
import torch.nn as nn

class MultimodalCNN(nn.Module):
    def __init__(self, num_struct_features, image_input_shape, num_classes):
        super(MultimodalCNN, self).__init__()

        # Structured Data Branch
        self.structured_branch = nn.Sequential(
            nn.Linear(num_struct_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Image Data Branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(in_channels=image_input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * (image_input_shape[1] // 4) * (image_input_shape[2] // 4), 64),
            nn.ReLU()
        )

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, structured_data, image_data):
        # Process each branch
        structured_out = self.structured_branch(structured_data)
        image_out = self.image_branch(image_data)

        # Concatenate outputs
        combined = torch.cat((structured_out, image_out), dim=1)

        # Final classification
        output = self.fusion(combined)
        return output