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

## Resultado 

Quando rodamos o programa de teste carregando a imagem do número 6, que está com o dígito bem centralizado e com as bordas grossas, conforme o dataset original, isso faz com que a CNN reconheça os padrões e acerte o valor de entrada.

Agora quando testamos inserido na entrada o número 2, ela erra, devido a imagem não parecer com o dataset, pois tem traços mais finos (apenas contorno), isso porque o dataset MNIST não generaliza bem para escrita livre.