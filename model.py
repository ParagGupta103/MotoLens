import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os


class CarClassifier(nn.Module):
    def __init__(self, num_classes=196):
        super(CarClassifier, self).__init__()
        # Load pre-trained ResNet-50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Fine-tuning strategy:
        # Freeze early layers, unfreeze 'layer4' and the final 'fc' layer.
        # This allows the model to learn high-level car features while keeping low-level edge detection.
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Replace the final fully connected layer to match the number of car classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), # Add dropout to prevent overfitting
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_data_loaders(data_dir, batch_size=32, input_size=448):
    # Advanced Data Augmentation for fine-grained classification
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.15)), # Resize slightly larger
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "test"]
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ["train", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device=None
):
    if device is None:
        device = torch.device("cpu")

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                # Optional: Save best model checkpoint here
                # torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Test Acc: {best_acc:4f}")
    return model


if __name__ == "__main__":
    DATA_DIR = "./data"
    BATCH_SIZE = 16 # Reduced batch size for higher resolution 448x448
    EPOCHS = 30
    INPUT_SIZE = 448

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}")

    try:
        loaders, sizes, classes = get_data_loaders(DATA_DIR, BATCH_SIZE, INPUT_SIZE)
        print(f"Detected {len(classes)} classes.")
    except FileNotFoundError:
        print(f"Error: Data directory not found at {DATA_DIR}.")
        exit(1)

    model = CarClassifier(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Differential Learning Rates:
    # Lower LR for pre-trained convolutional layers, higher LR for the new head.
    optimizer = optim.Adam([
        {'params': model.model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.model.fc.parameters(), 'lr': 1e-3}
    ])

    # Learning Rate Scheduler:
    # Decays the learning rate by a factor of 0.1 every 7 epochs.
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("Starting training with advanced features...")
    model = train_model(
        model, loaders, sizes, criterion, optimizer, step_lr_scheduler, num_epochs=EPOCHS, device=DEVICE
    )

    torch.save(model.state_dict(), "moto_lens_model_v2.pth")
    print("Model saved as moto_lens_model_v2.pth")
