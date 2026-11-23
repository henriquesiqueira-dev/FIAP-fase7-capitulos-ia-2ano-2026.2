import matplotlib.pyplot as plt
from skimage import data
import numpy as np

imagem_rgb = data.astronaut()# Carregando uma imagem RGB
print("Formato da imagem:", imagem_rgb.shape)

# Separando os canais R, G e B
canal_r = imagem_rgb[:, :, 0]
canal_g = imagem_rgb[:, :, 1]
canal_b = imagem_rgb[:, :, 2]

# Exibindo a imagem e os canais separadamente
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(imagem_rgb)
axs[0].set_title("Imagem RGB")
axs[0].axis('off')
axs[1].imshow(canal_r, cmap='grey')
axs[1].set_title("Canal Vermelho (R)")
axs[1].axis('off')
axs[2].imshow(canal_g, cmap='grey')
axs[2].set_title("Canal Verde (G)")
axs[2].axis('off')
axs[3].imshow(canal_b, cmap='grey')
axs[3].set_title("Canal Azul (B)")
axs[3].axis('off')
plt.show()

# Olhando as matrizes
print(imagem_rgb)
print(canal_r)
print(canal_g)
print(canal_b)
