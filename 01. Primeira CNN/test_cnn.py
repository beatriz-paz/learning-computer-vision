import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ==========================================
# IMPORTA OS MODELOS
# ==========================================

from cnn_mnist import SimpleCNN
from cnn_wave_mnist import WaveletCNN

# ==========================================
# MENU
# ==========================================

print("===================================")
print(" TESTE DE CNN - MNIST ")
print("===================================")
print("1 - CNN Simples")
print("2 - CNN com Wavelet")
print("===================================")

opcao = input("Escolha o modelo (1 ou 2): ")

if opcao == "1":
    model = SimpleCNN()
    model_path = "cnn_mnist.pth"
    print("\nModelo selecionado: CNN Simples")

elif opcao == "2":
    model = WaveletCNN()
    model_path = "wavelet_cnn_mnist.pth"
    print("\nModelo selecionado: CNN com Wavelet")

else:
    print("Opção inválida.")
    exit()

# ==========================================
# CARREGA PESOS
# ==========================================

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ==========================================
# ESCOLHA DA IMAGEM
# ==========================================

img_name = input("\nDigite o nome da imagem (ex: teste.png): ")
img_path = os.path.join("img", img_name)

if not os.path.exists(img_path):
    print("Imagem não encontrada!")
    exit()

# ==========================================
# PRÉ-PROCESSAMENTO DA IMAGEM
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(),      # garante 1 canal
    transforms.Resize((28, 28)), # ajusta tamanho
    transforms.ToTensor()
])

image = Image.open(img_path)
image = transform(image)
image = image.unsqueeze(0)  # (1, 1, 28, 28)

# ==========================================
# INFERÊNCIA
# ==========================================

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("\n================ RESULTADO ================")
print(f"Classe prevista: {predicted.item()}")
print("==========================================")
