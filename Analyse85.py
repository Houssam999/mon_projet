
Conversations

Non lus
 
1–1 sur 1
 

Autres messages
 
1–50 sur 8 694
 
53,44 Go utilisés sur 100 Go
Conditions d'utilisation · Confidentialité · Règlement du programme
Dernière activité sur le compte : il y a 2 minutes
Détails
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
        matrix = matrix.astype(data_type)
        self.matrix = matrix
        self.connectivity = connectivity
        # Création des décalages pour les voisins
        self.on_the_right = np.column_stack((matrix[:, 1:], matrix[:, -1]))
        self.on_the_left = np.column_stack((matrix[:, 0], matrix[:, :-1]))
        self.on_the_bot = np.row_stack((matrix[1:, :], matrix[-1, :]))
        self.on_the_top = np.row_stack((matrix[0, :], matrix[:-1, :]))
        if self.connectivity == 8:
            self.on_the_topleft = matrix[:-1, :-1]
            self.on_the_topright = matrix[:-1, 1:]
            self.on_the_botleft = matrix[1:, :-1]
            self.on_the_botright = matrix[1:, 1:]

            self.on_the_topleft = np.row_stack((self.on_the_topleft[0, :], self.on_the_topleft))
            self.on_the_topleft = np.column_stack((self.on_the_topleft[:, 0], self.on_the_topleft))

            self.on_the_topright = np.row_stack((self.on_the_topright[0, :], self.on_the_topright))
            self.on_the_topright = np.column_stack((self.on_the_topright, self.on_the_topright[:, -1]))

            self.on_the_botleft = np.row_stack((self.on_the_botleft, self.on_the_botleft[-1, :]))
            self.on_the_botleft = np.column_stack((self.on_the_botleft[:, 0], self.on_the_botleft))

            self.on_the_botright = np.row_stack((self.on_the_botright, self.on_the_botright[-1, :]))
            self.on_the_botright = np.column_stack((self.on_the_botright, self.on_the_botright[:, -1]))

    def is_equal(self, value, and_itself=False):
        if self.connectivity == 4:
            self.equal_neighbor_nb = np.dstack((np.equal(self.on_the_right, value),
                                                np.equal(self.on_the_left, value),
                                                np.equal(self.on_the_bot, value),
                                                np.equal(self.on_the_top, value)))
        elif self.connectivity == 8:
            self.equal_neighbor_nb = np.dstack((np.equal(self.on_the_right, value),
                                                np.equal(self.on_the_left, value),
                                                np.equal(self.on_the_bot, value),
                                                np.equal(self.on_the_top, value),
                                                np.equal(self.on_the_topleft, value),
                                                np.equal(self.on_the_topright, value),
                                                np.equal(self.on_the_botleft, value),
                                                np.equal(self.on_the_botright, value)))
        self.equal_neighbor_nb = np.sum(self.equal_neighbor_nb, 2, dtype=uint8)
        if and_itself:
            self.equal_neighbor_nb[np.not_equal(self.matrix, value)] = 0

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
            start_node = tuple(connected_node_coords[0])
            end_node = tuple(connected_node_coords[-1])
            segments.append((start_node, end_node, coords))
    return segments

def mesurer_largeur_segment(binary_image, segment):
    _, _, coords = segment
    distances = []
    for i in range(1, len(coords)-1):
        # Calculer le gradient pour obtenir la direction perpendiculaire
        dy = coords[i+1][0] - coords[i-1][0]
        dx = coords[i+1][1] - coords[i-1][1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        # Direction perpendiculaire
        perp_dx = -dy / norm
        perp_dy = dx / norm
        # Points de début et de fin pour le profil perpendiculaire
        r0 = coords[i][0] - perp_dy * 5  # Ajuster la longueur si nécessaire
        c0 = coords[i][1] - perp_dx * 5
        r1 = coords[i][0] + perp_dy * 5
        c1 = coords[i][1] + perp_dx * 5
        # Extraire le profil
        profile = profile_line(binary_image, (r0, c0), (r1, c1), order=0, mode='constant', cval=0)
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

def main():
    # Charger l'image et binariser
    data = np.load('network2.npy')
    binary_image = np.where(data[-1] == 2, 1, 0)

    # Squelettisation
    skeleton = morphology.skeletonize(binary_image)

    # Détection des nœuds
    nodes_centroid, node_positions = detect_nodes(skeleton)
    nodes_bool = nodes_centroid.astype(bool)
    node_positions_array = np.array(node_positions)

    # Trouver les segments
    segments = find_segments(skeleton, nodes_bool)

    # Calculer les degrés des nœuds
    node_degrees = extract_node_degrees(segments)

    # Créer un ensemble des positions des nœuds
    node_coords_set = set(map(tuple, node_positions_array))

    # Collecter tous les segments par combinaison de degrés
    segments_by_combination = {}
    counted_combinations = set()
    for segment in segments:
        node1 = segment[0]
        node2 = segment[1]
        degree1 = node_degrees.get(node1, 0)
        degree2 = node_degrees.get(node2, 0)
        combination = tuple(sorted((degree1, degree2)))
        coords = segment[2]
        # Vérifier s'il n'y a pas de nœuds intermédiaires (j'ai eu ce problème avec le segment 1 c'est pour ça que j'ai mis ça en place)
        intermediate_nodes = [tuple(coord) for coord in coords if tuple(coord) in node_coords_set and tuple(coord) != node1 and tuple(coord) != node2]
        if intermediate_nodes:
            continue  # Ignorer les segments avec des nœuds intermédiaires
        # Ajouter le segment à la liste correspondante si ce n'est pas déjà fait
        if combination not in segments_by_combination:
            segments_by_combination[combination] = (segment, degree1, degree2)
            counted_combinations.add(combination)
        # Si toutes les combinaisons possibles ont été trouvées, on peut arrêter
     
        if len(counted_combinations) == len(set(node_degrees.values()))**2:
            break

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

        # Mesurer les largeurs pour ce segment
        largeur_mesure = mesurer_largeur_segment(binary_image, segment)

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
                dy = coords[i+1][0] - coords[i-1][0]
                dx = coords[i+1][1] - coords[i-1][1]
                norm = np.hypot(dx, dy)
                if norm == 0:
                    continue
                perp_dx = -dy / norm
                perp_dy = dx / norm
                r0 = coords[i][0] - perp_dy * 5
                c0 = coords[i][1] - perp_dx * 5
                r1 = coords[i][0] + perp_dy * 5
                c1 = coords[i][1] + perp_dx * 5
                plt.plot([c0, c1], [r0, r1], 'blue', linewidth=0.5)
            plt.legend()
            plt.title(f'Segment {idx}: Squelette et lignes perpendiculaires')
            plt.axis('off')
            plt.savefig(f'segment_{idx}_image2.png', bbox_inches='tight')
            plt.show()
        else:
            print(f"Segment {idx}: Impossible de mesurer la largeur (segment trop court ou problème lors de la mesure)")

if __name__ == "__main__":
    main(