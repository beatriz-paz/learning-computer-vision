import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# -----------------------
# MESMO MODELO
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# CARREGAR MODELO TREINADO
# -----------------------
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_mnist.pth"))
model.eval()

# -----------------------
# TRANSFORMAÇÃO
# -----------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),      # <<< garante 28x28
    transforms.ToTensor(),
])

# -----------------------
# CARREGAR IMAGEM
# -----------------------
img = Image.open("img/num6_teste.png").convert("L")
#img = Image.open("img/num2_teste.png").convert("L")

# Aumentar contraste (acender o branco)
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(2.5)   # ajuste entre 2.0 e 3.0 se quiser
# Aplicar transformações
img = transform(img)
img = img.unsqueeze(0)

# -----------------------
# PREVISÃO
# -----------------------
with torch.no_grad():
    output = model(img)
    _, prediction = torch.max(output, 1)

print(f"Número previsto: {prediction.item()}")

plt.imshow(img[0][0], cmap="gray")
plt.title(f"Previsto: {prediction.item()}")
plt.axis("off")
plt.show()
