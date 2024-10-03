import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
from skimage.measure import profile_line
from scipy import ndimage
from numpy import uint8, int8
from cv2 import dilate
from collections import defaultdict

# Définition de la classe CompareNeighborsWithValue (algorithme 3)
class CompareNeighborsWithValue:
    def __init__(self, matrix, connectivity, data_type=int8):
        self.matrix = matrix.astype(data_type)
        self.connectivity = connectivity
        # Création des matrices décalées avec padding pour conserver les dimensions
        # Définition des dimensions
        shape = self.matrix.shape
        # Voisin de droite
        self.on_the_right = np.zeros_like(self.matrix)
        self.on_the_right[:, :-1] = self.matrix[:, 1:]
        # Voisin de gauche
        self.on_the_left = np.zeros_like(self.matrix)
        self.on_the_left[:, 1:] = self.matrix[:, :-1]
        # Voisin du bas
        self.on_the_bot = np.zeros_like(self.matrix)
        self.on_the_bot[:-1, :] = self.matrix[1:, :]
        # Voisin du haut
        self.on_the_top = np.zeros_like(self.matrix)
        self.on_the_top[1:, :] = self.matrix[:-1, :]
        if self.connectivity == 8:
            # Voisin en haut à gauche
            self.on_the_topleft = np.zeros_like(self.matrix)
            self.on_the_topleft[1:, 1:] = self.matrix[:-1, :-1]
            # Voisin en haut à droite
            self.on_the_topright = np.zeros_like(self.matrix)
            self.on_the_topright[1:, :-1] = self.matrix[:-1, 1:]
            # Voisin en bas à gauche
            self.on_the_botleft = np.zeros_like(self.matrix)
            self.on_the_botleft[:-1, 1:] = self.matrix[1:, :-1]
            # Voisin en bas à droite
            self.on_the_botright = np.zeros_like(self.matrix)
            self.on_the_botright[:-1, :-1] = self.matrix[1:, 1:]

    def is_equal(self, value, and_itself=False):
        neighbors = [
            self.on_the_right,
            self.on_the_left,
            self.on_the_bot,
            self.on_the_top
        ]
        if self.connectivity == 8:
            neighbors.extend([
                self.on_the_topleft,
                self.on_the_topright,
                self.on_the_botleft,
                self.on_the_botright
            ])
        self.equal_neighbor_nb = np.zeros_like(self.matrix, dtype=uint8)
        for neighbor in neighbors:
            self.equal_neighbor_nb += (neighbor == value).astype(uint8)
        if and_itself:
            self.equal_neighbor_nb *= (self.matrix == value).astype(uint8)

def detect_nodes(skeleton):
    cnv = CompareNeighborsWithValue(skeleton, 8)
    cnv.is_equal(1, and_itself=True)
    sure_terminations = np.zeros_like(skeleton, dtype=uint8)
    sure_terminations[cnv.equal_neighbor_nb == 1] = 1
    square_33 = np.ones((3, 3), dtype=uint8)
    nodes = sure_terminations.copy()
    for neighbor_nb in [8, 7, 6, 5, 4, 3]:
        potential_node = np.zeros_like(skeleton, dtype=uint8)
        potential_node[cnv.equal_neighbor_nb == neighbor_nb] = 1
        dilated_previous_intersections = dilate(nodes, square_33)
        potential_node *= (1 - dilated_previous_intersections)
        nodes[np.nonzero(potential_node)] = 1
    # Calcul des centroïdes des nœuds
    labeled_nodes, num_features = ndimage.label(nodes)
    node_positions = ndimage.center_of_mass(nodes, labeled_nodes, np.arange(1, num_features + 1))
    # Création d'une image des nœuds avec les centroïdes
    nodes_centroid = np.zeros_like(skeleton, dtype=int)
    for y, x in node_positions:
        nodes_centroid[int(round(y)), int(round(x))] = 1
    return nodes_centroid, node_positions

def find_segments(skeleton, nodes):
    skeleton_no_nodes = skeleton.copy()
    skeleton_no_nodes[nodes.astype(bool)] = 0
    num_labels, labels = cv2.connectedComponents(skeleton_no_nodes.astype(np.uint8))
    segments = []
    for label in range(1, num_labels):
        segment_mask = (labels == label)
        coords = np.column_stack(np.where(segment_mask))
        # Trouver les nœuds voisins du segment
        segment_dilated = morphology.binary_dilation(segment_mask, morphology.disk(1))
        connected_nodes = nodes.astype(bool) & segment_dilated
        connected_node_coords = np.column_stack(np.where(connected_nodes))
        if len(connected_node_coords) >= 2:
            # On prend les deux nœuds les plus éloignés
            dist_matrix = np.sum((connected_node_coords[:, np.newaxis] - connected_node_coords[np.newaxis, :]) ** 2, axis=2)
            idx_max = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            start_node = tuple(connected_node_coords[idx_max[0]])
            end_node = tuple(connected_node_coords[idx_max[1]])
            segments.append((start_node, end_node, coords))
        elif len(connected_node_coords) == 1:
            node1 = tuple(connected_node_coords[0])
            # Trouver l'extrémité opposée du segment
            distances = np.sum((coords - connected_node_coords[0]) ** 2, axis=1)
            idx_max = np.argmax(distances)
            node2 = tuple(coords[idx_max])
            segments.append((node1, node2, coords))
    return segments

def extract_node_degrees(segments):
    node_degrees = {}
    for segment in segments:
        node1 = segment[0]
        node2 = segment[1]
        for node in [node1, node2]:
            if node in node_degrees:
                node_degrees[node] += 1
            else:
                node_degrees[node] = 1
    return node_degrees

def mesurer_largeur_segment(binary_image, skeleton, segment, distance_map):
    _, _, coords = segment
    distances = []
    # Parcourir les points du segment en évitant les extrémités
    for i in range(1, len(coords)-1):
        y, x = coords[i]
        # Vérifier que le point est dans le masque de la veine
        if distance_map[y, x] == 0:
            continue
        # Calculer le gradient pour obtenir la direction perpendiculaire
        dy = coords[i+1][0] - coords[i-1][0]
        dx = coords[i+1][1] - coords[i-1][1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        # Direction perpendiculaire
        perp_dx = -dy / norm
        perp_dy = dx / norm
        # Longueur de la ligne perpendiculaire basée sur la distance au bord
        distance = distance_map[y, x]
        length = distance * 2
        if length < 1:
            continue
        # Points de début et de fin pour le profil perpendiculaire
        r0 = y - perp_dy * length / 2
        c0 = x - perp_dx * length / 2
        r1 = y + perp_dy * length / 2
        c1 = x + perp_dx * length / 2
        # Obtenir les coordonnées le long de la ligne de profil
        line_coords = np.linspace(0, 1, int(np.hypot(r1 - r0, c1 - c0)))
        rr = ((1 - line_coords) * r0 + line_coords * r1).astype(int)
        cc = ((1 - line_coords) * c0 + line_coords * c1).astype(int)
        # Filtrer les coordonnées en dehors de l'image
        valid_idx = (rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])
        rr = rr[valid_idx]
        cc = cc[valid_idx]
        # Vérifier si tous les points sont dans le masque de la veine du segment
        if not np.all(distance_map[rr, cc] > 0):
            continue
        # Extraire le profil
        profile = binary_image[rr, cc]
        width = np.sum(profile > 0)
        distances.append(width)
    if distances:
        largeurs = {
            'largeur_moyenne': np.mean(distances),
            'largeur_noeud_A': distances[0],
            'largeur_noeud_B': distances[-1],
            'largeur_milieu': distances[len(distances) // 2],
            'largeur_minimale': np.min(distances),
            'largeur_maximale': np.max(distances)
        }
        return largeurs
    else:
        return None  # Si aucune mesure n'a été effectuée

def main():
    # Charger l'image et binariser
    data = np.load('network2.npy')
    binary_image = np.where(data[-1] == 2, 1, 0).astype(bool)

    # Squelettisation
    skeleton = morphology.skeletonize(binary_image)

    # Détection des nœuds
    nodes_centroid, node_positions = detect_nodes(skeleton)
    nodes_bool = nodes_centroid.astype(bool)
    node_positions_array = np.array(node_positions, dtype=int)

    # Trouver les segments
    segments = find_segments(skeleton, nodes_bool)

    # Calculer les degrés des nœuds
    node_degrees = extract_node_degrees(segments)

    # Collecter tous les segments par combinaison de degrés
    segments_by_combination = {}
    for segment in segments:
        node1 = segment[0]
        node2 = segment[1]
        degree1 = node_degrees.get(node1, 1)
        degree2 = node_degrees.get(node2, 1)
        combination = tuple(sorted((degree1, degree2)))
        # Ajouter le segment à la liste correspondante si ce n'est pas déjà fait
        if combination not in segments_by_combination:
            segments_by_combination[combination] = (segment, degree1, degree2)

    # Afficher toutes les combinaisons détectées
    print(f"Combinaisons de degrés présentes dans le réseau : {sorted(segments_by_combination.keys())}")

    if not segments_by_combination:
        print("Aucun segment trouvé.")
        return

    # Pour chaque segment sélectionné, effectuer les mesures et les visualisations
    for idx, (combination, (segment, degree1, degree2)) in enumerate(segments_by_combination.items(), start=1):
        node1 = segment[0]
        node2 = segment[1]
        coords = segment[2]

        # Créer un masque du segment
        segment_mask = np.zeros_like(binary_image, dtype=bool)
        segment_mask[coords[:, 0], coords[:, 1]] = True
        # Créer un masque exclusif de la veine du segment
        vein_mask = morphology.binary_dilation(segment_mask, morphology.disk(3)) & binary_image
        # Calculer la carte de distance sur le masque de la veine
        distance_map = ndimage.distance_transform_edt(vein_mask)

        # Mesurer les largeurs pour ce segment
        largeur_mesure = mesurer_largeur_segment(binary_image, skeleton, segment, distance_map)

        if largeur_mesure:
            # Afficher les mesures
            print(f"\nSegment {idx}: Nœud A {node1} (degré {degree1}), Nœud B {node2} (degré {degree2})")
            print(f"Combinaison de degrés : {combination}")
            print(f"Largeur moyenne = {largeur_mesure['largeur_moyenne']:.2f} pixels")
            print(f"Largeur au nœud A = {largeur_mesure['largeur_noeud_A']} pixels")
            print(f"Largeur au nœud B = {largeur_mesure['largeur_noeud_B']} pixels")
            print(f"Largeur au milieu = {largeur_mesure['largeur_milieu']} pixels")
            print(f"Largeur minimale = {largeur_mesure['largeur_minimale']} pixels")
            print(f"Largeur maximale = {largeur_mesure['largeur_maximale']} pixels")

            # Image 1 : Squelette et réseau réel avec les nœuds
            plt.figure(figsize=(8, 8))
            plt.imshow(binary_image, cmap='gray')
            plt.imshow(skeleton, cmap='hot', alpha=0.5)
            plt.scatter([node1[1]], [node1[0]], c='green', s=50, label=f'Nœud A (degré {degree1})')
            plt.scatter([node2[1]], [node2[0]], c='yellow', s=50, label=f'Nœud B (degré {degree2})')
            plt.legend()
            plt.title(f'Segment {idx}: Squelette et réseau réel')
            plt.axis('off')
            plt.savefig(f'segment_{idx}_image1.png', bbox_inches='tight')
            plt.show()

            # Image 2 : Squelette et lignes perpendiculaires
            plt.figure(figsize=(8, 8))
            plt.imshow(skeleton, cmap='gray')
            plt.scatter([node1[1]], [node1[0]], c='green', s=50, label=f'Nœud A (degré {degree1})')
            plt.scatter([node2[1]], [node2[0]], c='yellow', s=50, label=f'Nœud B (degré {degree2})')
            # Tracer les lignes perpendiculaires utilisées pour mesurer la largeur
            for i in range(1, len(coords)-1, max(1, len(coords)//20)):
                y, x = coords[i]
                dy = coords[i+1][0] - coords[i-1][0]
                dx = coords[i+1][1] - coords[i-1][1]
                norm = np.hypot(dx, dy)
                if norm == 0:
                    continue
                perp_dx = -dy / norm
                perp_dy = dx / norm
                # Longueur de la ligne perpendiculaire basée sur la distance au bord
                if distance_map[y, x] == 0:
                    continue
                distance = distance_map[y, x]
                length = distance * 2
                if length < 1:
                    continue
                r0 = y - perp_dy * length / 2
                c0 = x - perp_dx * length / 2
                r1 = y + perp_dy * length / 2
                c1 = x + perp_dx * length / 2
                plt.plot([c0, c1], [r0, r1], 'blue', linewidth=0.5)
            plt.legend()
            plt.title(f'Segment {idx}: Squelette et lignes perpendiculaires')
            plt.axis('off')
            plt.savefig(f'segment_{idx}_image2.png', bbox_inches='tight')
            plt.show()
        else:
            print(f"Segment {idx}: Impossible de mesurer la largeur (segment trop court ou problème lors de la mesure)")

if __name__ == "__main__":
    main()