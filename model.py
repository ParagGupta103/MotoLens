import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

class CarClassifier(nn.Module):
    def __init__(self, num_classes=196):
        super(CarClassifier, self).__init__()
        # Load pre-trained ResNet-50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze initial layers for faster local training (Transfer Learning)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer to match the number of car classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def get_data_loaders(data_dir, batch_size=32):
    # Standard transforms for ResNet (224x224 input)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ImageFolder uses directory names as class labels automatically
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, device=None):
    if device is None:
        device = torch.device("cpu")
        
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = torch.tensor(0).to(device)

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

if __name__ == "__main__":
    # Local Configuration
    # Ensure this path points to where 'train' and 'test' folders are located
    DATA_DIR = './data'  
    BATCH_SIZE = 32
    EPOCHS = 5
    
    # Check for Apple Silicon (MPS), CUDA, or fallback to CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}")

    # 1. Initialize Data Loaders
    try:
        loaders, sizes, classes = get_data_loaders(DATA_DIR, BATCH_SIZE)
        print(f"Detected {len(classes)} classes.")
    except FileNotFoundError:
        print(f"Error: Data directory not found at {DATA_DIR}. Please check the path.")
        exit(1)

    # 2. Initialize Model
    model = CarClassifier(num_classes=len(classes)).to(DEVICE)
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters of the final layer (feature extraction)
    optimizer = optim.Adam(model.model.fc.parameters(), lr=0.001)

    # 4. Start Training
    print("Starting training...")
    model = train_model(model, loaders, sizes, criterion, optimizer, num_epochs=EPOCHS, device=DEVICE)
    
    # 5. Save the model state
    torch.save(model.state_dict(), 'moto_lens_model.pth')
    print("Model saved as moto_lens_model.pth")
