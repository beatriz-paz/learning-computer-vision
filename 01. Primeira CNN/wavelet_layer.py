import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletLayer(nn.Module):
    def __init__(self, in_channels=1, trainable=False):
        super().__init__()

        # ================================
        # Definição dos filtros Haar 2D
        # ================================
        # LL: aproximação (baixa frequência em ambas direções)
        ll = torch.tensor([[1, 1],
                           [1, 1]]) / 2

        # LH: detalhe vertical (baixa freq horizontal, alta vertical)
        lh = torch.tensor([[1, 1],
                           [-1, -1]]) / 2

        # HL: detalhe horizontal (alta freq horizontal, baixa vertical)
        hl = torch.tensor([[1, -1],
                           [1, -1]]) / 2

        # HH: detalhe diagonal (alta frequência em ambas direções)
        hh = torch.tensor([[1, -1],
                           [-1, 1]]) / 2

        # ================================
        # Empilhamento dos filtros
        # ================================
        # Junta os quatro filtros:
        # formato final: (4, 2, 2)
        filters = torch.stack([ll, lh, hl, hh])

        # Adiciona dimensão de canal de entrada
        # (4, 1, 2, 2)
        filters = filters.unsqueeze(1)

        # Repete os filtros para cada canal de entrada
        # Necessário quando in_channels > 1
        filters = filters.repeat(in_channels, 1, 1, 1)

        # Ajusta o formato final esperado pelo conv2d:
        # (4 * in_channels, 1, 2, 2)
        filters = filters.view(4 * in_channels, 1, 2, 2)

        # ================================
        # Define se os filtros são treináveis
        # ================================
        if trainable:
            # Filtros aprendidos via backpropagation
            self.filters = nn.Parameter(filters)
        else:
            # Filtros fixos (não participam do treinamento)
            self.register_buffer("filters", filters)

        # Armazena número de canais de entrada
        self.in_channels = in_channels

    def forward(self, x):
        # ================================
        # Convolução Wavelet
        # ================================
        # stride=2 realiza o downsampling (↓2)
        # groups=in_channels aplica os filtros
        # de forma independente em cada canal
        return F.conv2d(
            x,
            self.filters,
            stride=2,
            groups=self.in_channels
        )
