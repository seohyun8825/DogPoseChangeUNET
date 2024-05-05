import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DogKeypointDataset, transform
from model import UNet
import torch
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
def plot_images(original_img, target_keypoints, reconstructed_img, target_images, epoch):
    num_images = original_img.size(0)
    plt.figure(figsize=(20, num_images * 5))
    for idx in range(num_images):
        img_np = normalize_image(original_img[idx])
        img_recon = normalize_image(reconstructed_img[idx])
        img_target = normalize_image(target_images[idx])

        # Original image
        plt.subplot(num_images, 4, 4 * idx + 1)
        plt.title(f"Original Image {idx + 1}")
        plt.imshow(img_np.transpose(1, 2, 0))
        plt.axis('off')

        # Target keypoint
        blank_image = np.ones((256, 256, 3))
        plt.subplot(num_images, 4, 4 * idx + 2)
        plt.title(f"Target Keypoints {idx + 1}")
        plt.imshow(blank_image)
        x_points = target_keypoints[idx][0::2] * 256  
        y_points = target_keypoints[idx][1::2] * 256
        plt.scatter(x_points, y_points, color='blue', s=10)
        plt.axis('off')

        # Reconstructed image
        plt.subplot(num_images, 4, 4 * idx + 3)
        plt.title(f"Reconstructed Image {idx + 1}")
        plt.imshow(img_recon.transpose(1, 2, 0))
        plt.axis('off')

        # Target image
        plt.subplot(num_images, 4, 4 * idx + 4)
        plt.title(f"Target Image {idx + 1}")
        plt.imshow(img_target.transpose(1, 2, 0))
        plt.axis('off')

    plt.savefig(f'output_epoch_{epoch}.png')
    plt.close()



def normalize_image(image):
    image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min())  #  [0, 1]
    return image


def custom_collate_fn(batch):

    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])  
    return torch.utils.data.dataloader.default_collate(batch)
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DogKeypointDataset(
        json_file='C:/Users/user/Desktop/24-1/학회/APT-36k/AP-36k-patr1/apt36k_annotations.json',
        img_dir='C:/Users/user/Desktop/24-1/학회/APT-36k/AP-36k-patr1',
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    model = UNet(n_channels=3, n_keypoints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for images, target_keypoints, target_images in train_loader:
            images = images.to(device)
            target_images = target_images.to(device)
            target_keypoints = target_keypoints.to(device)  
            optimizer.zero_grad()

            with autocast():
                outputs = model(images, target_keypoints)  
                loss = criterion(outputs, target_images)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

        epoch_loss /= max(num_batches, 1)
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss}')

        if (epoch + 1) % 1 == 0: 
            with torch.no_grad():
                images = images.cpu().float()
                target_keypoints = target_keypoints.cpu().float()
                outputs = outputs.cpu().float()
                target_images = target_images.cpu().float()  
                plot_images(images, target_keypoints, outputs, target_images, epoch + 1)


if __name__ == '__main__':
    train_model()