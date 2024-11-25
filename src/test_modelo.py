import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# ====================================
# Parâmetros e Configurações
# ====================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

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
# Carregar Modelo Treinado
# ====================================
model = ASLConvNet().to(DEVICE)
model.load_state_dict(torch.load('asl_cnn.pth'))
model.eval()

# ====================================
# Avaliação do Modelo
# ====================================
# Carregar o conjunto de teste
test_dataset = ASLDataset(root_dir='C:/Users/bruno/Documents/GitHub/IA-Final/src/Dataset/archive/asl_alphabet_train/asl_alphabet_train', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# Avaliação
accuracy, all_preds, all_labels = evaluate(model, test_loader, DEVICE)
print(f'Acurácia no conjunto de teste: {accuracy:.2f}%')

# Relatório de Classificação e Matriz de Confusão
print("\nRelatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
plt.title('Matriz de Confusão')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Predito')
plt.show()
