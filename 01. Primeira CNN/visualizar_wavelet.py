import matplotlib.pyplot as plt

def plot_wavelet_subbands(wavelet_output):
    """
    wavelet_output: tensor (1, 4, H, W)
    """
    subbands = wavelet_output.squeeze(0).cpu()

    titles = ["LL (Aproximação)",
              "LH (Horizontal)",
              "HL (Vertical)",
              "HH (Detalhes)"]

    plt.figure(figsize=(8, 8))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(subbands[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
