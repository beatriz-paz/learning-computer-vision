import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from wavelet_layer import WaveletLayer  # Camada wavelet personalizada

# =======================================================
# DATASET MNIST
# =======================================================

# Transformação básica: converte imagem para tensor PyTorch
transform = transforms.Compose([
    transforms.ToTensor()
])

# Dataset de treino (MNIST)
train_dataset = datasets.MNIST(
    root="./data",        # Pasta onde os dados serão salvos
    train=True,           # Indica conjunto de treino
    download=True,        # Baixa se não existir
    transform=transform  # Aplica transformação
)

# Dataset de teste (MNIST)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,          # Indica conjunto de teste
    download=True,
    transform=transform
)

# DataLoader de treino: carrega os dados em batches
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # Tamanho do batch
    shuffle=True    # Embaralha os dados a cada época
)

# DataLoader de teste
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False   # Não precisa embaralhar no teste
)

# =======================================================
# DEFINIÇÃO DO MODELO CNN COM WAVELET NA PRIMEIRA CAMADA
# =======================================================

class WaveletCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ======================
        # Camada Wavelet
        # ======================
        # Aplica a transformada wavelet discreta 2D
        # Entrada: 1 canal (imagem em escala de cinza)
        # Saída: 4 canais (LL, LH, HL, HH)
        self.wavelet = WaveletLayer(
            in_channels=1,
            trainable=False   # Filtros wavelet fixos (não treináveis)
        )

        # ======================
        # CNN clássica
        # ======================
        # Após a wavelet, a entrada tem 4 canais
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Camada de pooling para reduzir a dimensionalidade espacial
        self.pool = nn.MaxPool2d(2, 2)

        # Dimensões ao longo da rede:
        # 28x28 → Wavelet → 14x14
        # Conv1 + Pool → 7x7
        # Conv2 + Pool → 3x3

        # Camadas totalmente conectadas (classificador)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (dígitos 0–9)

    def forward(self, x):
        # ======================
        # Passagem pela Wavelet
        # ======================
        x = self.wavelet(x)

        # ======================
        # Passagem pela CNN
        # ======================
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Achata o tensor para a camada totalmente conectada
        x = x.view(x.size(0), -1)

        # Camadas fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
# =======================================================
# TREINAMENTO DO MODELO
# =======================================================
if __name__ == "__main__":

    # Instancia o modelo
    model = WaveletCNN()

    # Função de perda para classificação multiclasse
    criterion = nn.CrossEntropyLoss()

    # Otimizador Adam
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    epochs = 5  # Número de épocas de treino

    for epoch in range(epochs):
        model.train()  # Coloca o modelo em modo treino
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Zera os gradientes

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("Treinamento finalizado!")

    # =======================================================
    # TESTE / AVALIAÇÃO
    # =======================================================

    model.eval()  # Coloca o modelo em modo avaliação

    correct = 0
    total = 0

    # Desativa cálculo de gradientes (economiza memória e tempo)
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            # Obtém a classe com maior probabilidade
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Cálculo da acurácia
    accuracy = 100 * correct / total
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")

    # Salva os pesos do modelo treinado
    torch.save(model.state_dict(), "wavelet_cnn_mnist.pth")
    print("Modelo salvo!")
