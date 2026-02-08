import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# =====================================================
# DEFINIÇÃO DO MESMO MODELO CNN USADO NO TREINAMENTO
# =====================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Primeira camada convolucional:
        # Entrada: 1 canal (imagem em tons de cinza)
        # Saída: 16 mapas de características
        # Kernel 3x3 com padding para manter o tamanho
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        # Segunda camada convolucional:
        # Entrada: 16 mapas
        # Saída: 32 mapas de características
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Camada de pooling:
        # Reduz largura e altura pela metade
        self.pool = nn.MaxPool2d(2, 2)

        # Camada totalmente conectada:
        # 32 canais * 7 * 7 (tamanho final após pooling)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Camada de saída:
        # 10 neurônios → classes de 0 a 9
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolução + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))

        # Segunda convolução + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))

        # Achata o tensor para passar à camada totalmente conectada
        x = x.view(x.size(0), -1)

        # Camada densa com ReLU
        x = torch.relu(self.fc1(x))

        # Camada final (logits)
        x = self.fc2(x)

        return x

# =====================================================
# CARREGAMENTO DO MODELO TREINADO
# =====================================================

# Cria a arquitetura do modelo
model = SimpleCNN()

# Carrega os pesos treinados salvos no arquivo
model.load_state_dict(torch.load("cnn_mnist.pth"))

# Coloca o modelo em modo de avaliação
# Desativa comportamentos de treino (ex: dropout)
model.eval()

# =====================================================
# TRANSFORMAÇÕES APLICADAS À IMAGEM DE TESTE
# =====================================================

transform = transforms.Compose([
    # Garante que a imagem tenha o tamanho esperado pela CNN
    transforms.Resize((28, 28)),

    # Converte a imagem PIL para tensor PyTorch
    # Valores passam para o intervalo [0, 1]
    transforms.ToTensor(),
])

# =====================================================
# CARREGAMENTO DA IMAGEM PARA TESTE
# =====================================================

# Abre a imagem do disco e converte para escala de cinza ("L")
print("Número da imagem de entrada: 6")
img = Image.open("img/num6_teste.png").convert("L")
# img = Image.open("img/num2_teste.png").convert("L")

# Aumenta o contraste da imagem
# Isso ajuda a deixar o dígito mais parecido com o MNIST
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(2.5)   # fator de contraste (ajustável)

# Aplica as transformações (resize + tensor)
img = transform(img)

# Adiciona a dimensão do batch
# De (1, 28, 28) → (1, 1, 28, 28)
img = img.unsqueeze(0)

# =====================================================
# PREVISÃO DO MODELO
# =====================================================

# Desativa cálculo de gradientes (inferência)
with torch.no_grad():
    # Passa a imagem pela rede
    output = model(img)

    # Seleciona a classe com maior valor de saída
    _, prediction = torch.max(output, 1)

# Exibe o número previsto pela CNN
print(f"Número previsto: {prediction.item()}")

# =====================================================
# VISUALIZAÇÃO DA IMAGEM E DO RESULTADO
# =======================
