import cv2
import numpy as np
from utilities import *
from sklearn.cluster import KMeans

# Variables globales
selected_points = []

# Charger une image (ou une image vierge si aucune n'est disponible)
texture_sol_base=cv2.imread('test_sol.jpg')
texture_sol=np.tile(texture_sol_base, (3, 3, 1))
print(texture_sol.shape)
image = np.ones(texture_sol.shape, dtype=np.uint8) * 255  # Une image blanche
cv2.putText(image, "Cliquez pour dessiner un chemin", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# Afficher l'image et capturer les clics
cv2.imshow("Selectionnez les points", image)
cv2.setMouseCallback("Selectionnez les points", select_points, {"points": selected_points})

print("Cliquez sur l'image pour sélectionner des points.")
print("Clic gauche : Ajouter un point, Clic droit : Supprimer un point.")
print("Appuyez sur 'q' pour terminer.")

while True:
    # Afficher les points sélectionnés sur l'image
    temp_image = image.copy()
    for x, y in selected_points:
        cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow("Selectionnez les points", temp_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quitter avec 'q'
        break

cv2.destroyAllWindows()

# Interpolation et affichage
if len(selected_points) > 1:
    # Choisir une méthode d'interpolation
    method = "spline"#input("Choisissez la méthode d'interpolation ('linear' ou 'spline') : ").strip().lower()

    try:
        x_interp, y_interp = interpolate_path(selected_points, method='spline')

        # Dessiner le chemin interpolé
        for (x, y) in zip(x_interp, y_interp):
            cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)

        # Afficher l'image finale
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Chemin interpolé ({method})")
        plt.show()

        # Sauvegarder les points sélectionnés
        # save_path(selected_points)
    except Exception as e:
        print(f"Erreur : {e}")
else:
    print("Pas assez de points sélectionnés pour interpoler un chemin.")

# Charger les images
image_chemin = image #cv2.imread('chemin.png', cv2.IMREAD_GRAYSCALE)
# texture_sol = cv2.imread('texture7.png')
image_path = 'texture_sol.png'  # Remplacez par le chemin de votre image
pierre = cv2.imread(image_path)
binary_pierre = apply_binary_segmentation(image_path)

segmented_texture = segment_texture_kmeans(pierre, n_clusters=2)
#segmented_image, segments = segment_texture_slic(texture_chemin)
#plt.imshow(segmented_texture)
#plt.title("Segmentation par (KMean)")
#plt.show()
spacing = 200
centers = get_path_centers(x_interp, y_interp, spacing)
tangents = compute_tangents(x_interp, y_interp, centers)
# Placer les pierres
result_with_stones = place_textures(texture_sol, pierre, binary_pierre, centers, tangents, scale=0.4)

plt.imshow(result_with_stones)
plt.title("Pierres placées sur le chemin")
plt.show()
