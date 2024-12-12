import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Charger une image de texture
image = cv2.imread('texture10.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))  # Redimensionner si nécessaire

def calculate_covariance_patches(image, patch_size=3):
    half_patch = patch_size // 2
    # Ajouter un padding à l'image
    padded_image = np.pad(image, pad_width=half_patch, mode='reflect')
    height, width = image.shape
    features = []

    for i in range(height):
        for j in range(width):
            # Extraire un patch centré sur le pixel
            patch = padded_image[i:i+patch_size, j:j+patch_size]
            # Calculer la matrice de covariance
            patch_vector = patch.flatten()
            covariance = np.cov(patch_vector, rowvar=False)
            features.append(covariance.flatten())  # Vectoriser la covariance
    return np.array(features)

features = calculate_covariance_patches(image)

# Appliquer k-means
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features)

# Reshape pour visualisation
segmented_image = labels.reshape(image.shape)
# Si le cluster blanc n'est pas le plus grand, inverser les labels
# Compter les pixels blancs et noirs
white_pixels = np.sum(segmented_image == 255)
black_pixels = np.sum(segmented_image == 0)

# Si le nombre de pixels blancs est inférieur à celui des pixels noirs, inverser les couleurs
if white_pixels < black_pixels:
    segmented_image = cv2.bitwise_not(segmented_image)
plt.imshow(segmented_image, cmap='gray')
plt.title("Segmented Image")
plt.show()