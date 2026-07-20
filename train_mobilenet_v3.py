import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import ssl

# Fix for the Windows SSL Certificate Download Issue
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# Configuration & Hyperparameters
# ==========================================
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001

# Update these paths to where your dataset is stored locally
TRAIN_PATH = "./Dataset/seg_train/seg_train"
TEST_PATH = "./Dataset/seg_test/seg_test"
SAVE_PATH = "./models"

def get_data_loaders(train_path, test_path, batch_size):
    # MobileNet V3 expects 224x224 inputs
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_train_dataset = ImageFolder(train_path, transform=train_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 is recommended for local Windows execution to avoid thread locking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Standard loss calculation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

def train_model(model, model_name, train_loader, val_loader, device, epochs=3, learning_rate=0.001):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0

    os.makedirs(SAVE_PATH, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"{model_name}.pth"))

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    return {"train_losses": train_losses, "train_accuracies": train_accuracies, "val_losses": val_losses, "val_accuracies": val_accuracies}

def plot_results(results, model_name):
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(results["train_losses"], label="Train Loss")
    plt.plot(results["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(results["train_accuracies"], label="Train Accuracy")
    plt.plot(results["val_accuracies"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(TRAIN_PATH, TEST_PATH, BATCH_SIZE)
    print("DataLoaders Ready!")

    # 2. Initialize MobileNet V3 Large
    mobilenet_v3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
    # Replace the final classification layer (index 3 in the classifier Sequential block)
    mobilenet_v3.classifier[3] = nn.Linear(in_features=1280, out_features=NUM_CLASSES)

    # 3. Train Model
    results = train_model(
        model=mobilenet_v3,
        model_name="MobileNetV3_Baseline",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # 4. Plot Metrics
    plot_results(results, "MobileNetV3")