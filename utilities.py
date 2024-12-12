import cv2
import numpy as np
import json
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import label2rgb
#from skimage.feature import greycomatrix, greycoprops
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.preprocessing.image import img_to_array, array_to_img




# =============================
#  Gestion des points interactifs
# =============================

def select_points(event, x, y, flags, param):
    """
    Callback pour gérer les clics de souris pour ajouter ou supprimer des points.
    
    - Clic gauche : Ajouter un point
    - Clic droit : Supprimer un point proche
    """
    selected_points = param['points']
    if event == cv2.EVENT_LBUTTONDOWN:  # Ajouter un point
        selected_points.append((x, y))
        print(f"Point ajouté : ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Supprimer un point
        for i, (px, py) in enumerate(selected_points):
            if abs(px - x) < 10 and abs(py - y) < 10:  # Si proche d'un point
                removed_point = selected_points.pop(i)
                print(f"Point supprimé : {removed_point}")
                break

# =============================
#  Gestion et interpolation des chemins
# =============================

def interpolate_path(points, num_points=500, method="linear"):
    """
    Interpole un chemin à partir des points donnés.
    
    Args:
        points (list): Liste des points (x, y).
        num_points (int): Nombre de points interpolés.
        method (str): Méthode d'interpolation ("linear" ou "spline").
    
    Returns:
        tuple: Coordonnées interpolées (x_interp, y_interp).
    """
    if len(points) < 2:
        raise ValueError("Au moins deux points sont nécessaires pour interpoler un chemin.")
    
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
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

def save_path(points, file_name="path.json"):
    """
    Sauvegarde les points dans un fichier JSON.
    
    Args:
        points (list): Liste des points (x, y).
        file_name (str): Nom du fichier.
    """
    with open(file_name, "w") as f:
        json.dump(points, f)
    print(f"Chemin sauvegardé dans {file_name}")

def load_path(file_name="path.json"):
    """
    Charge les points depuis un fichier JSON.
    
    Args:
        file_name (str): Nom du fichier.
    
    Returns:
        list: Liste des points (x, y).
    """
    try:
        with open(file_name, "r") as f:
            points = json.load(f)
        print(f"Chemin chargé depuis {file_name}")
        return points
    except FileNotFoundError:
        print(f"Erreur : fichier {file_name} introuvable.")
        return []


# =============================
#  Segmentation et filtrage
# =============================
def apply_gabor_filters(image, ksize=21, frequencies=[0.2, 0.5], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Applique des filtres de Gabor pour extraire des caractéristiques.
    
    Args:
        image (numpy array): Image en niveaux de gris.
        ksize (int): Taille du noyau.
        frequencies (list): Fréquences du filtre.
        orientations (list): Orientations du filtre.
    
    Returns:
        numpy array: Réponses des filtres empilées.
    """
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
    """
    Segmente une texture en utilisant k-means.
    
    Args:
        image (numpy array): Image en couleur.
        n_clusters (int): Nombre de clusters.
    
    Returns:
        numpy array: Image segmentée.
    """
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixel_values)
    segmented_image = labels.reshape(image.shape[:2])
    return segmented_image

def segment_texture_slic(image, n_segments=100):
    """
    Segmente une texture en utilisant SLIC (superpixels).
    
    Args:
        image (numpy array): Image en couleur.
        n_segments (int): Nombre de superpixels.
    
    Returns:
        tuple: Image segmentée, segments.
    """
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=1)
    segmented_image = label2rgb(segments, image, kind='avg')
    return segmented_image, segments

def apply_binary_segmentation(image_path, threshold_value=127):
    """
    Applique un seuillage binaire sur une image.
    
    Args:
        image_path (str): Chemin de l'image.
        threshold_value (int): Valeur de seuil.
    
    Returns:
        numpy array: Image binaire.
    """
    # Charger l'image en niveau de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer le seuillage pour obtenir une image binaire
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    # Compter les pixels blancs et noirs
    white_pixels = np.sum(binary_img == 255)
    black_pixels = np.sum(binary_img == 0)
    
    # Si le nombre de pixels blancs est inférieur à celui des pixels noirs, inverser les couleurs
    if white_pixels < black_pixels:
        binary_img = cv2.bitwise_not(binary_img)
    
    # Retourner l'image binaire
    return binary_img

# =============================
#  Placement et chevauchement
# =============================

def get_path_centers(x_path, y_path, spacing):
    """
    Calcule les centres équidistants le long d'un chemin.
    
    Args:
        x_path (list): Coordonnées X du chemin.
        y_path (list): Coordonnées Y du chemin.
        spacing (float): Espacement entre les centres.
    
    Returns:
        list: Liste des centres.
    """
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

def compute_tangents(x_path, y_path, centers):
    """
    Calcule les tangentes aux centres le long d'un chemin.
    
    Args:
        x_path (list): Coordonnées X du chemin.
        y_path (list): Coordonnées Y du chemin.
        centers (list): Liste des centres.
    
    Returns:
        list: Angles des tangentes.
    """
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
        tangents.append(np.pi / 2 - angle)  # Ajustement de l'orientation
    return tangents

def check_overlap(center, placed_centers, stone_radius):
    """
    Vérifie si une pierre chevauche les autres pierres placées en comparant les distances
    entre leurs centres. Si la distance est inférieure à un seuil, il y a chevauchement.

    Args:
        center (tuple): Le centre de la nouvelle pierre à placer.
        placed_centers (list): Liste des centres des pierres déjà placées.
        stone_radius (int): Rayon de la pierre.

    Returns:
        bool: True si la pierre chevauche, False sinon.
    """
    for placed_center in placed_centers:
        distance = np.sqrt((center[0] - placed_center[0]) ** 2 + (center[1] - placed_center[1]) ** 2)
        if distance < stone_radius * 2:  # Si la distance entre les centres est inférieure à deux rayons de pierre, il y a chevauchement
            return True
    return False

# Placer et orienter les pierres
def place_textures_with_overlap_check(background, stone_texture, white_mask, centers, tangents, scale=0.3, stone_radius=15):
    """
    Place et oriente les pierres le long d'un chemin, en évitant les chevauchements.
    """
    # Assurez-vous que les images sont en RGB pour Matplotlib et BGR pour OpenCV
    stone_texture_bgr = cv2.cvtColor(stone_texture, cv2.COLOR_RGB2BGR) if len(stone_texture.shape) == 3 else stone_texture
    background_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR) if len(background.shape) == 3 else background
    
    combined_image = background_bgr.copy()
    stone_resized = cv2.resize(stone_texture_bgr, 
                                (int(stone_texture_bgr.shape[1] * scale), int(stone_texture_bgr.shape[0] * scale)))
    # Redimensionner également le masque
    white_mask_resized = cv2.resize(white_mask, 
                                    (int(stone_texture_bgr.shape[1] * scale), int(stone_texture_bgr.shape[0] * scale)))

    placed_centers = []  # Liste pour suivre les centres des pierres déjà placées

    for (cx, cy), angle in zip(centers, tangents):
        # Vérifier si le centre chevauche avec un autre centre placé
        if check_overlap((cx, cy), placed_centers, stone_radius):
            continue  # Si chevauchement, sauter cette pierre
        
        # Ajouter ce centre à la liste des centres placés
        placed_centers.append((cx, cy))
        
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
        
        # Appliquer le masque pour ne placer que les parties blanches de la pierre
        for i in range(stone_rotated.shape[0]):
            for j in range(stone_rotated.shape[1]):
                if mask_rotated[i, j] > 0:  # Si le pixel est blanc
                    combined_image[y_start + i, x_start + j] = stone_rotated[i, j]
    
    # Convertir en RGB pour l'affichage
    return cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)