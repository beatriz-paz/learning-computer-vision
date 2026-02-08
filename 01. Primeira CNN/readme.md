# Minha primeira CNN

## Dataset

Para criação dessa CNN foi usado o dataset MNIST, que é um conjunto de imagens de números de 0 à 9 escritos à mão, contendo:

- 70.000 imagens no total
    - 60.000 para treino
    - 10.000 para teste

- Cada imagem é:
    - tamanho 28 × 28 pixels
    - preto e branco (1 canal)
    - um dígito de 0 a 9

## Execeução do programa

Para rodar o código, recomendo criar um ambiente virtual Python da seguinte forma:

```python
python -m venv venv
venv\Scripts\activate
```

Após, instalar as bibliotecas necessárias:

```python
pip install torch torchvision matplotlib

ou

pip install -r requirements.txt
```

Para rodar o programa:

```python
python cnn_mnist.py

e depois

python test_image.py
```

OBS.: primeiro executar a CNN para depois o programa de teste!

## CNN Simples

No programa python chamado __cnn_mnist.py__ foi desenvolvida uma Rede Neural Convolucional (CNN) em PyTorch para classificar imagens do conjunto MNIST, composto por dígitos manuscritos de 0 a 9 com tamanho 28×28 pixels em tons de cinza. Inicialmente, as imagens passam por um pré-processamento, onde são convertidas para tensores e normalizadas, facilitando o processo de aprendizado.

Os dados são organizados em mini-batches para tornar o treinamento mais eficiente e estável. A arquitetura da CNN possui duas camadas convolucionais, responsáveis pela extração de características das imagens, cada uma seguida pela função de ativação ReLU e por Max Pooling, que reduz a dimensionalidade e o custo computacional. Após essas etapas, as características extraídas são achatadas e processadas por uma camada totalmente conectada, culminando em uma camada de saída com 10 neurônios, correspondentes às classes dos dígitos.

O treinamento é realizado utilizando a função de perda CrossEntropyLoss e o otimizador Adam, ao longo de cinco épocas. Após o treinamento, o modelo é avaliado com um conjunto de teste independente, sendo calculada a acurácia para medir o desempenho da rede. Por fim, o modelo treinado é salvo como __cnn_mnist.pth__ para reutilização futura.

## CNN Wavelet

```
Imagem → Wavelet Decomposition → CNN → Classificador
(backprop passa por tudo)
```

DESCREVER O CÓDIGO


## Resultado 

### CNN Simples:

Quando rodamos o programa de teste carregando a imagem do número 6, que está com o dígito bem centralizado e com as bordas grossas, conforme o dataset original, isso faz com que a CNN reconheça os padrões e acerte o valor de entrada.

Agora quando testamos inserido na entrada o número 2, ela erra, devido a imagem não parecer com o dataset, pois tem traços mais finos (apenas contorno), isso porque o dataset MNIST não generaliza bem para escrita livre.

### CNN Wavelet: