import tkinter as tk
from tkinter import Canvas
from scipy.interpolate import splprep, splev
from scipy.ndimage import label
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt

class VueGrapique():
    """
    Classe qui permet à l'utilisateur de dessiner son chemin.
    """

    def __init__(self, width=512, height=512, epaisseur_chemin=5, nb_texture=8):
        """
        Constructeur de la classe.
        """
        self.width = width
        self.height = height
        self.epaisseur_chemin = epaisseur_chemin
        self.nb_texture = nb_texture

        self.points = []
        self.masque = np.zeros((self.height * nb_texture, self.width * nb_texture), dtype=np.uint8)

        # Initialisation de la fenêtre principale
        self.root = tk.Tk()
        self.root.title("Dessiner le chemin")

        # Initialisation du canvas
        self.canvas = Canvas(self.root, width=self.width, height=self.height, bg='black')
        self.canvas.pack()

        # Bind du clic gauche pour dessiner un point
        self.canvas.bind("<Button-1>", self.draw_point)

        # Création du bouton pour réinitialiser le chemin
        self.reset_button = tk.Button(self.root, text="Réinitialiser", command=self.reset_chemin)
        self.reset_button.pack(side=tk.LEFT)

        # Création du bouton pour valider le chemin
        self.validate_button = tk.Button(self.root, text="Valider", command=self.valider_chemin)
        self.validate_button.pack(side=tk.LEFT)


    def draw_point(self, event):
        """
        Permet de dessiner un point sur le canvas.
        """
        x = event.x
        y = event.y
        self.points.append((x, y))
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="white")
        self.update_courbe()

    def update_courbe(self):
        """
        Recalcule la courbe.
        """
        if len(self.points) < 4:
            return
        
        # Extraction des coordonnées
        x_coords, y_coords = zip(*self.points)

        # Générer une courbe spline lissée
        tck, u = splprep([x_coords, y_coords], s=2)
        u_fine = np.linspace(0, 1, 500)
        x_smooth, y_smooth = splev(u_fine, tck)

        # Effacer les courbes existantes
        self.canvas.delete("curve")

        # Tracé de la courbe
        for i in range(len(x_smooth) - 1):
            self.canvas.create_line(
                x_smooth[i], y_smooth[i],
                x_smooth[i+1], y_smooth[i+1],
                fill='red', width=2, tags="curve"
            )


    def reset_chemin(self):
        """
        Réinitialise le chemin.
        """
        self.points = []
        self.canvas.delete("all")


    def valider_chemin(self):
        """
        Valide le chemin.
        """
        # Mettre à jour le masque uniquement à la fin
        self.compute_masque()

        # Supprimer les boutons de l'interface graphique
        self.reset_button.destroy()
        self.validate_button.destroy()

        # Désactiver le clic gauche
        self.canvas.unbind("<Button-1>")

        # Afficher le masque pour validation
        self.afficher_masque()


    def compute_masque(self):
        """
        Calcule le masque avec la courbe lissée.
        """
        # Créer une image PIL pour dessiner des cercles
        mask_image = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask_image)

        # Dessiner des cercles autour de chaque point de la courbe
        x_coords, y_coords = zip(*self.points)
        tck, u = splprep([x_coords, y_coords], s=2)
        u_fine = np.linspace(0, 1, 500)
        x_smooth, y_smooth = splev(u_fine, tck)

        for x, y in zip(x_smooth, y_smooth):
            x, y = int(round(x)), int(round(y))
            draw.ellipse(
                [x - self.epaisseur_chemin, y - self.epaisseur_chemin, x + self.epaisseur_chemin, y + self.epaisseur_chemin],
                fill=255
            )
        
        # Convertir l'image PIL en masque numpy
        chemin_masque = np.array(mask_image) // 255

        # Redimensionner le masque pour le masque de textures
        masque_redim = Image.fromarray(chemin_masque).resize((self.width * self.nb_texture, self.height * self.nb_texture), Image.NEAREST)
        self.masque = np.array(masque_redim)


    def afficher_masque(self):
        """
        Ouvre la fenêtre de validation du masque pour que l'utilisateur puisse le vérifier.
        """
        mask_image = Image.fromarray(self.masque * 255)  # Convertir en image PIL
        self.mask_photo = ImageTk.PhotoImage(mask_image)

        # Affichage du masque dans la fenêtre principale
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.mask_photo)

        # Message d'attente pour l'utilisateur
        self.message_label = tk.Label(self.root, text="Appuyez sur une touche pour continuer")
        self.message_label.pack(side=tk.BOTTOM)

        # Bloquer l'exécution jusqu'à ce qu'une touche soit pressée
        self.root.bind("<KeyPress>", self.continuer)


    def continuer(self, event):
        """
        Permet de continuer après avoir validé le masque.
        """
        # Libère le bind
        self.root.unbind("<KeyPress>") 
        # Supprimer le message d'attente
        self.message_label.destroy()
        # Fermer la fenêtre
        self.root.quit()


    def tracer_chemin(self):
        """
        Lance l'interface graphique pour tracer le chemin.
        Return: le masque du chemin
        """
        self.root.mainloop()
        return self.masque


    def charger_masque_textures(self, file_path):
        """
        Charge un masque de textures à partir d'un fichier.
        """
        # Charger l'image en niveaux de gris
        image = Image.open(file_path).convert("L")

        # Redimensionner l'image
        image = image.resize((self.width, self.height), Image.NEAREST)

        # Convertir l'image en masque binaire
        masque = np.array(image) // 255

        # Créer un masque de textures
        masque_textures = np.zeros((self.height * self.nb_texture, self.width * self.nb_texture), dtype=np.uint8)

        # Remplir le masque de textures
        for i in range(self.nb_texture):
            for j in range(self.nb_texture):
                masque_textures[i*self.height:(i+1)*self.height, j*self.width:(j+1)*self.width] = masque
        
        # # Test de l'affichage avec matplotlib
        # plt.imshow(masque_textures, cmap='gray')
        # plt.show()

        return masque_textures
        

    def appliquer_masque_textures(self, masque_textures):
        """
        Applique le masque de textures sur le masque du chemin.
        """
        

# Exemple d'utilisation
vue = VueGrapique()
masque = vue.tracer_chemin()
masque_textures = vue.charger_masque_textures("./Textures/texture3_masque.png")
# masque_combine = masque * masque_textures
labeled_mask, num_labels = label(masque_textures)
masque_final = np.zeros_like(masque_textures)

for label_id in range(1, num_labels + 1):
    indices_zone = np.where(labeled_mask == label_id)
    for i,j in zip(*indices_zone):
        if masque[i,j] == 1:
            masque_final[indices_zone] = 1
            break
            


plt.imshow(masque_final, cmap='gray')
plt.show()