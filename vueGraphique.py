import tkinter as tk
from tkinter import Canvas
from scipy.interpolate import splprep, splev
import numpy as np
from PIL import Image, ImageTk, ImageDraw

class VueGrapique():
    """
    Classe qui permet à l'utilisateur de dessiner son chemin.
    """

    def __init__(self, width=800, height=500, epaisseur_chemin=30):
        """
        Constructeur de la classe.
        """
        self.width = width
        self.height = height
        self.epaisseur_chemin = epaisseur_chemin

        self.points = []
        self.masque = np.zeros((self.height, self.width), dtype=np.uint8)

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
        self.masque[:] = np.array(mask_image) // 255


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

vue = VueGrapique()
test = vue.tracer_chemin()
print(test)
