import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import MultimodalCNN

def evaluate_features(X_num, X_img, y, device, num_epochs=100, lr=0.001):
    """
    Evaluate feature selection using MultimodalCNN as the fitness function.

    Args:
        X_num (torch.Tensor): Structured data.
        X_img (torch.Tensor): Image data.
        y (torch.Tensor): Labels.
        device (str): Device to run the model on.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        float: Final model accuracy on the dataset.
    """
    num_struct_features = X_num.shape[1]
    image_input_shape = X_img.shape[1:]  # C, H, W
    num_classes = len(torch.unique(y))

    # Initialize model
    model = MultimodalCNN(num_struct_features, image_input_shape, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_num, X_img)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Calculate accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X_num, X_img)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()

    return accuracy