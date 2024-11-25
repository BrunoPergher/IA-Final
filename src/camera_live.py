import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2

# ====================================
# Parâmetros e Configurações
# ====================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# Transformações da Imagem
# ====================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ====================================
# Acessando a Câmera em Tempo Real para Predição
# ====================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocessamento da imagem capturada
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predição
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = chr(predicted.item() + ord('A'))
    
    # Mostrar o resultado na janela da câmera
    cv2.putText(frame, f'Predicted: {label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Recognition', frame)
    
    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
