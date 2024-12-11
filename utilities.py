import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import label2rgb
#from skimage.feature import greycomatrix, greycoprops
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.preprocessing.image import img_to_array, array_to_img




# Fonction de gestion des clics (ajout/suppression de points)
def select_points(event, x, y, flags, param):
    """Callback pour gérer les clics de souris pour ajouter ou supprimer des points."""
    selected_points = param['points']
    if event == cv2.EVENT_LBUTTONDOWN:  # Clic gauche pour ajouter un point
        selected_points.append((x, y))
        print(f"Point ajouté : ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Clic droit pour supprimer un point
        for i, (px, py) in enumerate(selected_points):
            if abs(px - x) < 10 and abs(py - y) < 10:  # Si proche d'un point existant
                removed_point = selected_points.pop(i)
                print(f"Point supprimé : {removed_point}")
                break


# Fonction pour interpoler un chemin
def interpolate_path(points, num_points=500, method="linear"):
    """Interpole un chemin à partir des points avec méthode choisie (linéaire ou spline)."""
    if len(points) < 2:
        raise ValueError("Au moins deux points sont nécessaires pour interpoler un chemin.")
    
    # Extraire les coordonnées X et Y
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # Calculer une interpolation
    t = np.arange(len(points))
    t_interp = np.linspace(0, len(points) - 1, num_points)

    if method == "linear":
        x_interp = interp1d(t, x, kind='linear')(t_interp)
        y_interp = interp1d(t, y, kind='linear')(t_interp)
    elif method == "spline":
        x_interp = interp1d(t, x, kind='cubic')(t_interp)
        y_interp = interp1d(t, y, kind='cubic')(t_interp)
    else:
        raise ValueError("Méthode non supportée : choisissez 'linear' ou 'spline'.")

    return x_interp, y_interp


# Fonction pour sauvegarder les points dans un fichier JSON
def save_path(points, file_name="path.json"):
    """Sauvegarder les points sélectionnés dans un fichier JSON."""
    with open(file_name, "w") as f:
        json.dump(points, f)
    print(f"Chemin sauvegardé dans {file_name}")


# Fonction pour charger les points depuis un fichier JSON
def load_path(file_name="path.json"):
    """Charger des points depuis un fichier JSON."""
    try:
        with open(file_name, "r") as f:
            points = json.load(f)
        print(f"Chemin chargé depuis {file_name}")
        return points
    except FileNotFoundError:
        print(f"Erreur : fichier {file_name} introuvable.")
        return []

# pour la segmentation   
def apply_gabor_filters(image, ksize=21, frequencies=[0.2, 0.5], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    responses = []
    for freq in frequencies:
        for theta in orientations:
            kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, freq * ksize, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(filtered)
    return np.stack(responses, axis=-1)
# Redimensionner les textures pour correspondre à l'image du chemin
#texture_sol = cv2.resize(texture_sol, (image_chemin.shape[1], image_chemin.shape[0]))
#texture_chemin = cv2.resize(texture_chemin, (image_chemin.shape[1], image_chemin.shape[0]))
# Convertir l'image en niveaux de gris
#texture_chemin_gray = cv2.cvtColor(texture_chemin, cv2.COLOR_BGR2GRAY)
#gabor_features = apply_gabor_filters(texture_chemin_gray)
# Redimensionner les réponses pour faire correspondre chaque pixel à ses caractéristiques de Gabor
#print("Forme de texture_chemin :", texture_chemin_gray.shape)
#height, width = texture_chemin_gray.shape
#responses_reshaped = gabor_features.reshape(-1, gabor_features.shape[-1])  # Mettre en forme (nombre_pixels, nb_features)

# Appliquer k-means pour segmenter l'image en 2 groupes (ou plus, selon la complexité de la texture)
#kmeans = KMeans(n_clusters=2, random_state=42)
#labels = kmeans.fit_predict(responses_reshaped)  # Labels pour chaque pixel

# Reshaper les labels pour avoir une image segmentée
#segmented_image = labels.reshape(height, width)

def segment_texture_kmeans(image, n_clusters=2):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixel_values)
    
    segmented_image = labels.reshape(image.shape[:2])
    return segmented_image

def segment_texture_slic(image, n_segments=100):
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=1)
    segmented_image = label2rgb(segments, image, kind='avg')
    return segmented_image, segments
def apply_binary_segmentation(image_path, threshold_value=127):
    # Charger l'image en niveau de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer le seuillage pour obtenir une image binaire
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Retourner l'image binaire
    return binary_img

# Placement des centres le long du chemin
def get_path_centers(x_path, y_path, spacing):
    centers = []
    current_distance = 0
    for i in range(1, len(x_path)):
        dx = x_path[i] - x_path[i-1]
        dy = y_path[i] - y_path[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        current_distance += distance
        
        if current_distance >= spacing:
            centers.append((x_path[i], y_path[i]))
            current_distance = 0
    return centers

# Calcul de la tangente pour chaque pierre
def compute_tangents(x_path, y_path, centers):
    tangents = []
    for cx, cy in centers:
        closest_idx = np.argmin([(cx - x)**2 + (cy - y)**2 for x, y in zip(x_path, y_path)])
        if closest_idx < len(x_path) - 1:
            dx = x_path[closest_idx + 1] - x_path[closest_idx]
            dy = y_path[closest_idx + 1] - y_path[closest_idx]
        else:
            dx = x_path[closest_idx] - x_path[closest_idx - 1]
            dy = y_path[closest_idx] - y_path[closest_idx - 1]
        angle = np.arctan2(dy, dx)
        tangents.append(angle)
    return tangents

# Placer et orienter les pierres
def place_textures(background, stone_texture, white_mask, centers, tangents, scale=0.3):
    combined_image = background.copy()
    stone_resized = cv2.resize(stone_texture, 
                                (int(stone_texture.shape[1] * scale), int(stone_texture.shape[0] * scale)))
    # Redimensionner également le masque
    white_mask_resized = cv2.resize(white_mask, 
                                    (int(stone_texture.shape[1] * scale), int(stone_texture.shape[0] * scale)))

    for (cx, cy), angle in zip(centers, tangents):
        # Créer une matrice de rotation
        rotation_matrix = cv2.getRotationMatrix2D((stone_resized.shape[1] // 2, stone_resized.shape[0] // 2), 
                                                  np.degrees(angle), 1.0)
        stone_rotated = cv2.warpAffine(stone_resized, rotation_matrix, 
                                       (stone_resized.shape[1], stone_resized.shape[0]))
        mask_rotated = cv2.warpAffine(white_mask_resized, rotation_matrix, 
                                      (stone_resized.shape[1], stone_resized.shape[0]))
        
        # Déterminer la position pour placer la pierre
        x_start = int(cx - stone_rotated.shape[1] // 2)
        y_start = int(cy - stone_rotated.shape[0] // 2)
        x_end = x_start + stone_rotated.shape[1]
        y_end = y_start + stone_rotated.shape[0]
        
        # Vérifier les limites de l'image
        if x_start < 0 or y_start < 0 or x_end > combined_image.shape[1] or y_end > combined_image.shape[0]:
            continue
        
        # Placer la pierre sur l'image
        #combined_image[y_start:y_end, x_start:x_end] = stone_rotated
        # Appliquer le masque pour ne placer que les parties blanches de la pierre
        for i in range(stone_rotated.shape[0]):
            for j in range(stone_rotated.shape[1]):
                if mask_rotated[i, j] > 0:  # Si le pixel est blanc
                    combined_image[y_start + i, x_start + j] = stone_rotated[i, j]
    
    
    return combined_image