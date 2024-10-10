import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
from scipy import ndimage
from collections import defaultdict
from numpy import uint8, int8

# Définition de la classe CompareNeighborsWithValue (algorithme 3)
class CompareNeighborsWithValue:
    def __init__(self, matrix, connectivity, data_type=int8):
        self.matrix = matrix.astype(data_type)
        self.connectivity = connectivity
        # Création des matrices décalées avec padding pour conserver les dimensions
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
    # Nous calculons le nombre de voisins pour chaque pixel du squelette
    cnv = CompareNeighborsWithValue(skeleton, 8)
    cnv.is_equal(1, and_itself=True)
    neighbor_counts = cnv.equal_neighbor_nb

    # Identification des nœuds : pixels avec 1 ou plus de 2 voisins
    nodes = ((neighbor_counts == 1) | (neighbor_counts > 2)) & skeleton

    # Étiquetage des nœuds
    labeled_nodes, num_labels = ndimage.label(nodes, structure=np.ones((3, 3), dtype=np.uint8))

    # Positions des nœuds
    node_positions = ndimage.center_of_mass(nodes, labeled_nodes, range(1, num_labels + 1))
    # Conversion des positions en entiers
    node_positions = [tuple(map(int, pos)) for pos in node_positions]
    # Création d'un mapping label -> position
    label_to_position = {label: pos for label, pos in zip(range(1, num_labels + 1), node_positions)}

    return labeled_nodes, label_to_position

def find_segments(skeleton, labeled_nodes, label_to_position):
    # Suppression des nœuds du squelette
    skeleton_wo_nodes = skeleton.copy()
    skeleton_wo_nodes[labeled_nodes > 0] = 0

    # Détection des segments (composantes connectées sans les nœuds)
    num_labels, labels = cv2.connectedComponents(skeleton_wo_nodes.astype(np.uint8))
    segments = []

    for label in range(1, num_labels):
        segment_mask = (labels == label)
        coords = np.column_stack(np.where(segment_mask))

        # Dilatation du segment pour trouver les nœuds adjacents
        dilated_segment = morphology.binary_dilation(segment_mask, morphology.disk(2))
        overlapping_nodes = labeled_nodes * dilated_segment
        node_labels = np.unique(overlapping_nodes[overlapping_nodes > 0])

        if len(node_labels) >= 2:
            # Si au moins deux nœuds sont connectés, on trouve les deux plus éloignés
            node_positions = [label_to_position[n_label] for n_label in node_labels]
            distances = np.sum((np.array(node_positions)[:, None] - np.array(node_positions)[None, :]) ** 2, axis=2)
            idx_max = np.unravel_index(np.argmax(distances), distances.shape)
            start_label = node_labels[idx_max[0]]
            end_label = node_labels[idx_max[1]]
            start_pos = label_to_position[start_label]
            end_pos = label_to_position[end_label]
            segments.append({
                'start_label': start_label,
                'end_label': end_label,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'coords': coords
            })
        elif len(node_labels) == 1:
            # Si un seul nœud est connecté, on trouve l'extrémité la plus éloignée
            start_label = node_labels[0]
            start_pos = label_to_position[start_label]
            distances = np.sum((coords - np.array(start_pos)) ** 2, axis=1)
            idx_max = np.argmax(distances)
            end_pos = tuple(coords[idx_max])
            segments.append({
                'start_label': start_label,
                'end_label': None,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'coords': coords
            })
    return segments

def extract_node_degrees(segments):
    node_degrees = defaultdict(int)
    for segment in segments:
        start_label = segment['start_label']
        end_label = segment['end_label']
        node_degrees[start_label] += 1
        if end_label is not None:
            node_degrees[end_label] += 1
    return node_degrees

def mesurer_largeur_segment(binary_image, segment, distance_map):
    coords = segment['coords']
    distances = []
    for i in range(1, len(coords) -1):
        y, x = coords[i]
        if distance_map[y, x] == 0:
            continue
        dy = coords[i+1][0] - coords[i-1][0]
        dx = coords[i+1][1] - coords[i-1][1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        perp_dx = -dy / norm
        perp_dy = dx / norm
        length = distance_map[y, x] * 2
        if length < 1:
            continue
        r0 = y - perp_dy * length / 2
        c0 = x - perp_dx * length / 2
        r1 = y + perp_dy * length / 2
        c1 = x + perp_dx * length / 2
        line_length = int(np.hypot(r1 - r0, c1 - c0))
        if line_length == 0:
            continue
        line_coords = np.linspace(0, 1, line_length)
        rr = ((1 - line_coords) * r0 + line_coords * r1).astype(int)
        cc = ((1 - line_coords) * c0 + line_coords * c1).astype(int)
        valid_idx = (rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])
        rr = rr[valid_idx]
        cc = cc[valid_idx]
        if not np.all(binary_image[rr, cc]):
            continue
        width = len(rr)
        distances.append(width)
    if distances:
        largeurs = {
            'largeur_moyenne': np.mean(distances),
            'largeur_noeud_A': distances[0],
            'largeur_noeud_B': distances[-1],
            'largeur_milieu': distances[len(distances)//2],
            'largeur_minimale': np.min(distances),
            'largeur_maximale': np.max(distances)
        }
        return largeurs
    else:
        return None

def main():
    # Charger l'image et binariser
    data = np.load('network2.npy')
    binary_image = (data[-1] == 2).astype(bool)

    # Squelettisation
    skeleton = morphology.skeletonize(binary_image)

    # Détection des nœuds
    labeled_nodes, label_to_position = detect_nodes(skeleton)

    # Trouver les segments
    segments = find_segments(skeleton, labeled_nodes, label_to_position)

    # Calculer les degrés des nœuds
    node_degrees = extract_node_degrees(segments)

    # Définir les combinaisons intéressantes
    combinaisons_interessees = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    # Regrouper les segments par combinaison de degrés
    segments_by_combination = defaultdict(list)
    for segment in segments:
        start_label = segment['start_label']
        end_label = segment['end_label']
        degree1 = node_degrees.get(start_label, 0)
        if end_label is not None:
            degree2 = node_degrees.get(end_label, 0)
        else:
            degree2 = 0
        combination = tuple(sorted((degree1, degree2)))
        if combination in combinaisons_interessees:
            segments_by_combination[combination].append(segment)

    # Afficher les combinaisons trouvées
    print(f"Combinaisons de degrés présentes dans le réseau : {sorted(segments_by_combination.keys())}")

    if not segments_by_combination:
        print("Aucun segment trouvé correspondant aux combinaisons spécifiées.")
        return

    idx = 1
    for combination, segment_list in segments_by_combination.items():
        # Sélectionner un seul segment par combinaison
        segment = next(iter(segment_list))
        start_label = segment['start_label']
        end_label = segment['end_label']
        start_pos = label_to_position[start_label]
        degree1 = node_degrees.get(start_label, 0)
        if end_label is not None:
            end_pos = label_to_position[end_label]
            degree2 = node_degrees.get(end_label, 0)
        else:
            end_pos = segment['end_pos']
            degree2 = 0
        coords = segment['coords']

        # Créer un masque du segment
        segment_mask = np.zeros_like(binary_image, dtype=bool)
        segment_mask[coords[:, 0], coords[:, 1]] = True
        vein_mask = morphology.binary_dilation(segment_mask, morphology.disk(3)) & binary_image
        distance_map = ndimage.distance_transform_edt(vein_mask)

        # Mesurer la largeur du segment
        largeur_mesure = mesurer_largeur_segment(binary_image, segment, distance_map)

        if largeur_mesure:
            # Afficher les mesures
            print(f"\nSegment {idx}: Nœud A {start_pos} (degré {degree1}), Nœud B {end_pos} (degré {degree2})")
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
            plt.scatter([start_pos[1]], [start_pos[0]], c='green', s=50, label=f'Nœud A (degré {degree1})')
            plt.scatter([end_pos[1]], [end_pos[0]], c='yellow', s=50, label=f'Nœud B (degré {degree2})')
            plt.legend()
            plt.title(f'Segment {idx}: Squelette et réseau réel')
            plt.axis('off')
            plt.savefig(f'segment_{idx}_image1.png', bbox_inches='tight')
            plt.show()

            # Image 2 : Squelette et lignes perpendiculaires
            plt.figure(figsize=(8, 8))
            plt.imshow(skeleton, cmap='gray')
            plt.scatter([start_pos[1]], [start_pos[0]], c='green', s=50, label=f'Nœud A (degré {degree1})')
            plt.scatter([end_pos[1]], [end_pos[0]], c='yellow', s=50, label=f'Nœud B (degré {degree2})')
            # Tracer les lignes perpendiculaires
            for i in range(1, len(coords) - 1, max(1, len(coords) // 20)):
                y, x = coords[i]
                dy = coords[i+1][0] - coords[i-1][0]
                dx = coords[i+1][1] - coords[i-1][1]
                norm = np.hypot(dx, dy)
                if norm == 0:
                    continue
                perp_dx = -dy / norm
                perp_dy = dx / norm
                if distance_map[y, x] == 0:
                    continue
                length = distance_map[y, x] * 2
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
            print(f"Segment {idx}: Impossible de mesurer la largeur")
        idx +=1

if __name__ == "__main__":
    main()
