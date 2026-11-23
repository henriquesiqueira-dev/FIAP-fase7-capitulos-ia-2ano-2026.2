import cv2
import matplotlib.pyplot as plt
from skimage import data, util, filters

imagem_original = data.camera()

# Adicionando Ruído sal-e-pimenta na imagem
imagem_ruidosa = util.random_noise(
    imagem_original,
    mode='s&p',
    amount=0.05
)

imagem_ruidosa_uint8 = (imagem_ruidosa * 255).astype('uint8')

# -------------------------------------------------
# Filtros passa-baixa
# -------------------------------------------------
# Filtro da média
filtro_media = cv2.blur(imagem_ruidosa_uint8, (5, 5))

# Filtro de mediana
filtro_mediana = cv2.medianBlur(imagem_ruidosa_uint8, 5)

# -------------------------------------------------
# Filtro passa-alta (Sobel)
# -------------------------------------------------
# Sobel direto na imagem com ruído
sobel_ruido = filters.sobel(imagem_ruidosa_uint8)

# Sobel após filtro de mediana
sobel_mediana = filters.sobel(filtro_mediana)

# -------------------------------------------------
# Visualização
# -------------------------------------------------
plt.figure(figsize=(8, 6))
plt.subplot(2, 3, 1)
plt.title("Imagem Original")
plt.imshow(imagem_original, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.title("Imagem com Ruído (S&P)")
plt.imshow(imagem_ruidosa, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.title("Filtro da Média")
plt.imshow(filtro_media, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.title("Filtro de Mediana")
plt.imshow(filtro_mediana, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.title("Sobel (com Ruído)")
plt.imshow(sobel_ruido, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.title("Sobel após Mediana")
plt.imshow(sobel_mediana, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
