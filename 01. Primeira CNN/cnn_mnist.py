import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# =====================================================
# PRÉ-PROCESSAMENTO DOS DADOS
# =====================================================

# Pipeline de transformações aplicado às imagens do MNIST
transform = transforms.Compose([
    # Converte a imagem PIL para tensor PyTorch
    # Resultado: tensor com valores no intervalo [0, 1]
    transforms.ToTensor(),

    # Normaliza os pixels:
    # média = 0.5 e desvio padrão = 0.5
    # Isso faz os valores ficarem aproximadamente em [-1, 1]
    transforms.Normalize((0.5,), (0.5,))
])

# Carrega o conjunto de treinamento do MNIST: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
train_dataset = datasets.MNIST(
    root="./data",        # Pasta onde os dados serão salvos
    train=True,           # Indica que é o conjunto de treino
    download=True,        # Faz download se não existir
    transform=transform  # Aplica o pré-processamento definido acima
)

# DataLoader organiza os dados em mini-batches (pequeno grupo de amostras)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # Número de imagens por batch
    shuffle=True    # Embaralha os dados a cada época
)

# =====================================================
# DEFINIÇÃO DO MODELO CNN
# =====================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Primeira camada convolucional:
        # - Entrada: 1 canal (imagem em tons de cinza)
        # - Saída: 16 mapas de características
        # - Kernel 3x3
        # - Padding=1 mantém o tamanho espacial
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        # Segunda camada convolucional:
        # - Entrada: 16 canais
        # - Saída: 32 mapas de características
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Camada de pooling:
        # Reduz a dimensão espacial pela metade (2x2)
        self.pool = nn.MaxPool2d(2, 2)

        # Camada totalmente conectada:
        # Após duas operações de pooling, a imagem 28x28 vira 7x7
        # 32 canais * 7 * 7 = 1568 neurônios de entrada
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Camada de saída:
        # 128 neurônios → 10 classes (dígitos de 0 a 9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Aplica convolução + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))

        # Segunda convolução + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))

        # "Achata" o tensor para formato (batch_size, features)
        x = x.view(x.size(0), -1)

        # Camada totalmente conectada com ReLU
        x = torch.relu(self.fc1(x))

        # Camada de saída (sem softmax)
        # CrossEntropyLoss aplica o softmax internamente
        x = self.fc2(x)

        return x

# =====================================================
# TREINAMENTO DO MODELO
# =====================================================

# Instancia o modelo
model = SimpleCNN()

# Função de perda:
# CrossEntropyLoss combina LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()

# Otimizador Adam para atualizar os pesos da rede
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Número de épocas de treinamento
epochs = 5

# Loop de treinamento
for epoch in range(epochs):
    running_loss = 0.0  # Acumula a perda da época

    # Itera sobre os batches do DataLoader
    for images, labels in train_loader:
        # Zera os gradientes acumulados
        optimizer.zero_grad()

        # Forward pass: calcula as saídas da rede
        outputs = model(images)

        # Calcula a perda comparando saída e rótulos
        loss = criterion(outputs, labels)

        # Backpropagation: calcula os gradientes
        loss.backward()

        # Atualiza os pesos da rede
        optimizer.step()

        # Soma a perda do batch
        running_loss += loss.item()

    # Exibe a perda média da época
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

print("Treinamento finalizado!")

# =====================================================
# TESTE - AVALIAÇÃO DE ACURÁCIA
# =====================================================

# Carrega o conjunto de teste do MNIST
test_dataset = datasets.MNIST(
    root="./data",
    train=False,          # Indica conjunto de teste
    download=True,
    transform=transform
)

# DataLoader para o conjunto de teste
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False         # Não precisa embaralhar no teste
)

# Coloca o modelo em modo avaliação
# Desativa comportamentos como dropout e batchnorm
model.eval()

correct = 0  # Número de previsões corretas
total = 0    # Total de amostras avaliadas

# Desativa o cálculo de gradientes (economiza memória)
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass
        outputs = model(images)

        # Obtém a classe com maior probabilidade
        _, predicted = torch.max(outputs, 1)

        # Atualiza o total de amostras
        total += labels.size(0)

        # Conta quantas previsões estão corretas
        correct += (predicted == labels).sum().item()

# Calcula a acurácia em porcentagem
accuracy = 100 * correct / total
print(f"Acurácia no teste: {accuracy:.2f}%")

# =====================================================
# SALVAMENTO DO MODELO
# =====================================================

# Salva apenas os pesos treinados da rede
torch.save(model.state_dict(), "cnn_mnist.pth")
print("Modelo salvo em cnn_mnist.pth")
