import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure

# Carrega imagem RGB
imagem = data.astronaut().astype(np.float32) / 255.0

# Brilho
brilho = np.clip(imagem + 0.2, 0, 1)

# Contraste
contraste = np.clip(imagem * 1.5, 0, 1)

# Normalização
normalizada = (imagem - imagem.min(axis=(0,1))) / (imagem.max(axis=(0,1)) - imagem.min(axis=(0,1)))

# Negativo
negativo = 1.0 - imagem

# Limiarização
intensidade = imagem.mean(axis=2)
limiar = intensidade > 0.5

# Correção gama
gamma = exposure.adjust_gamma(imagem, gamma=2)

# Quantização (redução de níveis)
niveis = 4
quantizada = np.floor(imagem * niveis) / niveis

# Exibição
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs[0,0].imshow(imagem)
axs[0,0].set_title("Original (RGB)")
axs[0,1].imshow(brilho)
axs[0,1].set_title("Brilho")
axs[0,2].imshow(contraste)
axs[0,2].set_title("Contraste")
axs[0,3].imshow(normalizada)
axs[0,3].set_title("Normalização")
axs[1,0].imshow(negativo)
axs[1,0].set_title("Negativo")
axs[1,1].imshow(limiar, cmap='gray')
axs[1,1].set_title("Limiarização")
axs[1,2].imshow(gamma)
axs[1,2].set_title("Correção Gama")
axs[1,3].imshow(quantizada)
axs[1,3].set_title("Quantização")
for ax in axs.flat:
    ax.axis('off')
plt.tight_layout()
plt.show()
