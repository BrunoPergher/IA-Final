import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ====================================
# Parâmetros e Configurações
# ====================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# Dataset Personalizado para ASL
# ====================================
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {letter: idx for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        
        for label in os.listdir(root_dir):
            if label in self.label_map:  # Certifique-se de que o rótulo está no mapa de letras
                letter_dir = os.path.join(root_dir, label)
                if os.path.isdir(letter_dir):
                    for img_name in os.listdir(letter_dir):
                        self.image_paths.append(os.path.join(letter_dir, img_name))
                        self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ====================================
# Transformações de Data Augmentation
# ====================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ====================================
# Carregando Datasets
# ====================================
train_dataset = ASLDataset(root_dir='C:/Users/bruno/Documents/GitHub/IA-Final/src/Dataset/archive/asl_alphabet_train/asl_alphabet_train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================================
# Definição do Modelo
# ====================================
class ASLConvNet(nn.Module):
    def __init__(self):
        super(ASLConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 26)  # 26 classes (A-Z)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ====================================
# Inicialização do Modelo, Função de Custo e Otimizador
# ====================================
model = ASLConvNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ====================================
# Funções de Treinamento e Avaliação
# ====================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        progress_bar.set_postfix({"Loss": running_loss / (progress_bar.n + 1)})
    
    return running_loss / len(train_loader)

# ====================================
# Treinamento do Modelo
# ====================================
train_losses = []
for epoch in range(NUM_EPOCHS):
    loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    train_losses.append(loss)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss:.4f}')

# ====================================
# Salvar o Modelo Treinado
# ====================================
torch.save(model.state_dict(), 'asl_cnn.pth')

# ====================================
# Visualização do Treinamento
# ====================================
plt.plot(train_losses)
plt.title('Custo de Treinamento')
plt.xlabel('Época')
plt.ylabel('Custo (loss)')
plt.show()
