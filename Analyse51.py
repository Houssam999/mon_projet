import numpy as np
import cv2
from skimage import morphology
from skimage.draw import line
import pandas as pd

def neighbor_count(matrix):
    padded = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)
    neighbors = np.zeros(matrix.shape, dtype=int)
    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            if padded[i, j] == 1:
                neighbors[i-1, j-1] = np.sum(padded[i-1:i+2, j-1:j+2]) - 1
    return neighbors

def detect_nodes(skeleton):
    neighbors = neighbor_count(skeleton)
    nodes = np.zeros(skeleton.shape, dtype=int)
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 1 and neighbors[i, j] in [3, 4, 5, 6, 7, 8]:
                nodes[i, j] = 1
    return nodes

def find_segments(binary_image, nodes):
    skeleton_no_nodes = binary_image.copy()
    skeleton_no_nodes[nodes == 1] = 0

    num_labels, labels = cv2.connectedComponents(skeleton_no_nodes.astype(np.uint8))

    segments = []
    for label in range(1, num_labels):
        segment_mask = (labels == label)
        coords = np.column_stack(np.where(segment_mask))
        segment_dilated = morphology.binary_dilation(segment_mask, morphology.disk(1))
        node_coords = [(r, c) for r, c in np.column_stack(np.where(nodes)) if segment_dilated[r, c]]
        if len(node_coords) >= 2:
            segments.append((node_coords[0], node_coords[-1], coords))
    return segments

def mesurer_largeur_veines(binary_image, segments):
    all_largeurs = []
    for segment in segments:
        (r1, c1), (r2, c2), coords = segment
        distances = []
        for r, c in coords:
            rr, cc = line(r - 10, c, r + 10, c)
            profile = binary_image[rr, cc]
            width = np.sum(profile)
            distances.append(width)
        largeurs = {
            'largeur_moyenne': np.mean(distances),
            'largeur_noeud_A': distances[0] if distances else 0,
            'largeur_noeud_B': distances[-1] if distances else 0,
            'largeur_milieu': distances[len(distances) // 2] if distances else 0,
            'largeur_minimale': np.min(distances) if distances else 0,
            'largeur_maximale': np.max(distances) if distances else 0
        }
        all_largeurs.append(largeurs)
    return all_largeurs

def extract_graph(skeleton, nodes):
    """
    Extrait un graphe du squelette en utilisant les nœuds détectés et les segments.
    Renvoie une liste des arêtes avec leurs longueurs.
    """
    edges = []
    segments = find_segments(skeleton, nodes)
    for (r1, c1), (r2, c2), coords in segments:
        # Ajouter l'arête avec les nœuds connectés et la longueur du segment
        length = np.linalg.norm([r2 - r1, c2 - c1])
        edges.append(((r1, c1), (r2, c2), length))
    return edges

def main():
    # Charger l'image et binariser
    data = np.load('network2.npy')
    binary_image = np.where(data[-1] == 2, 1, 0)

    # Squelettiser l'image
    skeleton = morphology.skeletonize(binary_image)

    # Détecter les nœuds
    nodes = detect_nodes(skeleton)

    # Trouver les segments de veine entre les nœuds
    segments = find_segments(skeleton, nodes)

    # Mesurer la largeur des veines
    all_largeurs = mesurer_largeur_veines(binary_image, segments)
    
    # Extraire le graphe à partir du squelette
    edges = extract_graph(skeleton, nodes)
    print("Graphe extrait avec les arêtes :")
    for edge in edges:
        print(f"Nœud {edge[0]} connecté à {edge[1]} avec une longueur de {edge[2]:.2f} pixels")

    # Afficher les explications
    print("Explication du code :")
    print("""
1. Détection des nœuds :
    - Utilisation de la méthode neighbor_count pour compter les voisins de chaque pixel et identifier les nœuds en fonction du nombre de voisins.
    - Le code supprime les nœuds détectés du squelette pour obtenir des segments distincts.
2. Détection des segments de veine :
    - Utilisation de cv2.connectedComponents pour étiqueter les segments.
    - Les segments sont dilatés pour retrouver les nœuds connectés.
3. Mesure de la largeur des veines :
    - Pour chaque segment de veine, des lignes perpendiculaires sont tracées aux coordonnées intermédiaires.
    - Les profils de pixels actifs (pixels de veine) sont mesurés pour obtenir la largeur de la veine à chaque point.
    - Les métriques de largeur (moyenne, au nœud A, au nœud B, au milieu, minimale, maximale) sont calculées pour chaque segment.
4. Extraction du graphe :
    - Les segments détectés sont utilisés pour connecter les nœuds entre eux avec des arêtes pondérées par la distance entre les nœuds.
    - Le graphe est ensuite affiché avec les arêtes et leurs longueurs.
5. Affichage des largeurs mesurées dans un tableau :
    - Utilisation de pandas pour créer un tableau récapitulatif listant chaque segment de veine avec les différentes largeurs mesurées.
    """)
    
    # Créer un DataFrame et sauvegarder dans un fichier CSV
    df = pd.DataFrame(all_largeurs)
    df.to_csv('largeurs_veines.csv', index=False)
    print("Tableau des largeurs de veines sauvegardé dans 'largeurs_veines.csv'")

if __name__ == "__main__":
    main()
