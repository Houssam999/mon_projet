#!/usr/bin/env python3
"""
This script contains the class for detecting networks from grayscale images of Physarum polycephalum.
"""

import cv2
import numpy as np
from numpy import uint8, int8, int32
from skimage import morphology
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt


class CompareNeighborsWithValue:
    def __init__(self, matrix, connectivity, data_type=int8):
        self.matrix = matrix.astype(data_type)
        self.connectivity = connectivity
        # Create shifted matrices with padding to maintain dimensions
        # Right neighbor
        self.on_the_right = np.zeros_like(self.matrix)
        self.on_the_right[:, :-1] = self.matrix[:, 1:]
        # Left neighbor
        self.on_the_left = np.zeros_like(self.matrix)
        self.on_the_left[:, 1:] = self.matrix[:, :-1]
        # Bottom neighbor
        self.on_the_bottom = np.zeros_like(self.matrix)
        self.on_the_bottom[:-1, :] = self.matrix[1:, :]
        # Top neighbor
        self.on_the_top = np.zeros_like(self.matrix)
        self.on_the_top[1:, :] = self.matrix[:-1, :]
        if self.connectivity == 8:
            # Top-left neighbor
            self.on_the_topleft = np.zeros_like(self.matrix)
            self.on_the_topleft[1:, 1:] = self.matrix[:-1, :-1]
            # Top-right neighbor
            self.on_the_topright = np.zeros_like(self.matrix)
            self.on_the_topright[1:, :-1] = self.matrix[:-1, 1:]
            # Bottom-left neighbor
            self.on_the_bottomleft = np.zeros_like(self.matrix)
            self.on_the_bottomleft[:-1, 1:] = self.matrix[1:, :-1]
            # Bottom-right neighbor
            self.on_the_bottomright = np.zeros_like(self.matrix)
            self.on_the_bottomright[:-1, :-1] = self.matrix[1:, 1:]

    def is_equal(self, value, and_itself=False):
        neighbors = [
            self.on_the_right,
            self.on_the_left,
            self.on_the_bottom,
            self.on_the_top
        ]
        if self.connectivity == 8:
            neighbors.extend([
                self.on_the_topleft,
                self.on_the_topright,
                self.on_the_bottomleft,
                self.on_the_bottomright
            ])
        self.equal_neighbor_nb = np.zeros_like(self.matrix, dtype=np.uint8)
        for neighbor in neighbors:
            self.equal_neighbor_nb += (neighbor == value).astype(np.uint8)
        if and_itself:
            self.equal_neighbor_nb *= (self.matrix == value).astype(np.uint8)


class NetworkDetection:
    def __init__(self, lighter_background):
        """
        :param lighter_background: True if the background of the image is lighter than the network to detect
        :type lighter_background: bool
        """
        self.lighter_background = lighter_background

    def skeletonize(self):
        """
        Perform skeletonization of the network.
        """
        self.skeleton = morphology.skeletonize(self.network)

    def detect_nodes(self):
        """
        Detect nodes in the skeletonized network.
        """
        # Calculate the number of neighbors for each pixel in the skeleton
        cnv = CompareNeighborsWithValue(self.skeleton, 8)
        cnv.is_equal(1, and_itself=True)
        neighbor_counts = cnv.equal_neighbor_nb

        # Identification of nodes: pixels with 1 or more than 2 neighbors
        nodes = ((neighbor_counts == 1) | (neighbor_counts > 2)) & self.skeleton

        # Labeling nodes
        labeled_nodes, num_labels = ndimage.label(nodes, structure=np.ones((3, 3), dtype=np.uint8))
        # Ensure labels are int32 to handle more than 255 labels
        self.labeled_nodes = labeled_nodes.astype(np.int32)

        # Node positions
        node_positions = ndimage.center_of_mass(nodes, self.labeled_nodes, range(1, num_labels + 1))
        node_positions = [tuple(map(int, pos)) for pos in node_positions]
        self.label_to_position = {label: pos for label, pos in zip(range(1, num_labels + 1), node_positions)}

    def find_segments(self):
        """
        Find segments in the skeletonized network.
        """
        # Remove nodes from the skeleton
        skeleton_wo_nodes = self.skeleton.copy()
        skeleton_wo_nodes[self.labeled_nodes > 0] = 0

        # Detection of segments (connected components without nodes)
        num_labels, labels = cv2.connectedComponents(skeleton_wo_nodes.astype(np.uint8))
        # Ensure labels are int32 to handle more than 255 labels
        labels = labels.astype(np.int32)
        self.segments = []

        for label in range(1, num_labels):
            segment_mask = (labels == label)
            coords = np.column_stack(np.where(segment_mask))

            # Dilate the segment to find adjacent nodes
            dilated_segment = morphology.binary_dilation(segment_mask, morphology.disk(2))
            overlapping_nodes = self.labeled_nodes * dilated_segment
            node_labels = np.unique(overlapping_nodes[overlapping_nodes > 0])

            if len(node_labels) >= 2:
                # If at least two nodes are connected, find the two farthest apart
                node_positions = [self.label_to_position[n_label] for n_label in node_labels]
                distances = np.sum((np.array(node_positions)[:, None] - np.array(node_positions)[None, :]) ** 2, axis=2)
                idx_max = np.unravel_index(np.argmax(distances), distances.shape)
                start_label = node_labels[idx_max[0]]
                end_label = node_labels[idx_max[1]]
                start_pos = self.label_to_position[start_label]
                end_pos = self.label_to_position[end_label]
                self.segments.append({
                    'start_label': start_label,
                    'end_label': end_label,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'coords': coords
                })
            elif len(node_labels) == 1:
                # If only one node is connected, find the farthest extremity
                start_label = node_labels[0]
                start_pos = self.label_to_position[start_label]
                distances = np.sum((coords - np.array(start_pos)) ** 2, axis=1)
                idx_max = np.argmax(distances)
                end_pos = tuple(coords[idx_max])
                self.segments.append({
                    'start_label': start_label,
                    'end_label': None,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'coords': coords
                })

    def get_segment_width(self, binary_image, segment, distance_map):
        """
        Measure the width of a segment.
        """
        coords = segment['coords']
        distances = []
        epsilon = 1e-6  # To avoid division by zero
        min_length = 3   # Minimum length of the perpendicular line
        coords_length = len(coords)
        for i in range(coords_length):
            if coords_length == 1:
                y, x = coords[i]
                dy, dx = 0, 0
            elif i == 0:
                y, x = coords[i]
                dy = coords[i + 1][0] - y
                dx = coords[i + 1][1] - x
            elif i == coords_length - 1:
                y, x = coords[i]
                dy = y - coords[i - 1][0]
                dx = x - coords[i - 1][1]
            else:
                y, x = coords[i]
                dy = coords[i + 1][0] - coords[i - 1][0]
                dx = coords[i + 1][1] - coords[i - 1][1]
            norm = np.hypot(dx, dy)
            if norm < epsilon:
                continue
            perp_dx = -dy / norm
            perp_dy = dx / norm
            if distance_map[y, x] == 0:
                continue
            length = max(distance_map[y, x] * 2, min_length)
            r0 = y - perp_dy * length / 2
            c0 = x - perp_dx * length / 2
            r1 = y + perp_dy * length / 2
            c1 = x + perp_dx * length / 2
            # Ensure indices are within image bounds
            r0 = np.clip(r0, 0, binary_image.shape[0] - 1)
            c0 = np.clip(c0, 0, binary_image.shape[1] - 1)
            r1 = np.clip(r1, 0, binary_image.shape[0] - 1)
            c1 = np.clip(c1, 0, binary_image.shape[1] - 1)
            line_length = int(np.hypot(r1 - r0, c1 - c0))
            if line_length == 0:
                continue
            line_coords = np.linspace(0, 1, line_length)
            rr = ((1 - line_coords) * r0 + line_coords * r1).astype(int)
            cc = ((1 - line_coords) * c0 + line_coords * c1).astype(int)
            # Remove duplicates
            unique_points = set(zip(rr, cc))
            if unique_points:
                rr, cc = zip(*unique_points)
                rr = np.array(rr)
                cc = np.array(cc)
                # Avoid out-of-bounds indices after removing duplicates
                valid_idx = (rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])
                rr = rr[valid_idx]
                cc = cc[valid_idx]
                # Relax the pixel inclusion criterion
                vein_pixels = binary_image[rr, cc]
                vein_pixel_ratio = np.sum(vein_pixels) / len(vein_pixels)
                if vein_pixel_ratio < 0.5:  # Require at least 50% vein pixels
                    continue
                width = len(rr)
                distances.append(width)
        if distances:
            widths = {
                'average_width': np.mean(distances),
                'width_node_A': distances[0],
                'width_node_B': distances[-1],
                'middle_width': distances[len(distances) // 2],
                'minimum_width': np.min(distances),
                'maximum_width': np.max(distances)
            }
            return widths
        else:
            # I use that as a formality precaution: if no measurements could be made, estimate width from the distance map
            median_distance = np.median(distance_map[coords[:, 0], coords[:, 1]])
            if median_distance > 0:
                estimated_width = median_distance * 2
                widths = {
                    'average_width': estimated_width,
                    'width_node_A': estimated_width,
                    'width_node_B': estimated_width,
                    'middle_width': estimated_width,
                    'minimum_width': estimated_width,
                    'maximum_width': estimated_width
                }
                return widths
            else:
                return None

    def networks_detection(self, data):
        """
        Analyze the network over a time-lapse sequence.

        :param data: 3D numpy array containing the time-lapse images (time, height, width)
        :type data: numpy.ndarray
        """
        num_frames = data.shape[0]
        last_frame = num_frames - 1

        # Initialize DataFrames to store information about vertices and edges
        vertices_df = pd.DataFrame(columns=['label', 't_start', 't_end', 'y', 'x'])
        edges_df = pd.DataFrame(columns=['label', 't_start', 't_end', 'start_y', 'start_x',
                                         'end_y', 'end_x', 'length', 'average_width',
                                         'width_node_A', 'width_node_B', 'middle_width',
                                         'minimum_width', 'maximum_width'])

        next_vertex_label = 1  # Identifier for vertices (nodes)
        next_edge_label = 1    # Identifier for edges

        # Previous labels for nodes and edges
        previous_vertex_labels = None
        previous_edge_labels = None

        # For tracking t_start and t_end of vertices and edges
        vertex_times = pd.DataFrame(columns=['label', 't_start', 't_end'])
        edge_times = pd.DataFrame(columns=['label', 't_start', 't_end'])

        for t in range(num_frames):
            print(f"Processing frame {t}/{last_frame}")

            # Update binary image
            binary_image = (data[t] == 2).astype(bool)
            self.network = binary_image
            self.skeletonize()
            self.detect_nodes()
            self.find_segments()

            # Label the current nodes
            current_vertex_labels = self.labeled_nodes.copy()
            num_nodes = np.max(current_vertex_labels)

            # If previous labels exist, assign labels based on overlap
            if previous_vertex_labels is not None:
                # Compute overlap between current labels and previous labels
                overlap = previous_vertex_labels * (current_vertex_labels > 0)
                for label in range(1, num_nodes + 1):
                    mask = (current_vertex_labels == label)
                    overlapping_labels = np.unique(overlap[mask])
                    overlapping_labels = overlapping_labels[overlapping_labels > 0]
                    if overlapping_labels.size > 0:
                        # Assign the most frequent overlapping label
                        assigned_label = np.bincount(overlapping_labels).argmax()
                        current_vertex_labels[mask] = assigned_label
                        # Update t_end in vertex_times
                        vertex_times.loc[vertex_times['label'] == assigned_label, 't_end'] = t
                    else:
                        # Assign a new label
                        assigned_label = next_vertex_label
                        current_vertex_labels[mask] = assigned_label
                        vertex_times = vertex_times.append({'label': assigned_label, 't_start': t, 't_end': t},
                                                           ignore_index=True)
                        next_vertex_label += 1
            else:
                # First frame, assign labels as is
                for label in range(1, num_nodes + 1):
                    assigned_label = next_vertex_label
                    mask = (current_vertex_labels == label)
                    current_vertex_labels[mask] = assigned_label
                    vertex_times = vertex_times.append({'label': assigned_label, 't_start': t, 't_end': t},
                                                       ignore_index=True)
                    next_vertex_label += 1

            # Update vertices_df
            for label in np.unique(current_vertex_labels):
                if label == 0:
                    continue
                if not (vertices_df['label'] == label).any():
                    # Get position
                    position = np.mean(np.column_stack(np.where(current_vertex_labels == label)), axis=0).astype(int)
                    vertex_data = {
                        'label': label,
                        't_start': int(vertex_times.loc[vertex_times['label'] == label, 't_start']),
                        't_end': int(vertex_times.loc[vertex_times['label'] == label, 't_end']),
                        'y': position[0],
                        'x': position[1]
                    }
                    vertices_df = vertices_df.append(vertex_data, ignore_index=True)
                else:
                    # Update t_end
                    vertices_df.loc[vertices_df['label'] == label, 't_end'] = int(
                        vertex_times.loc[vertex_times['label'] == label, 't_end'])

            previous_vertex_labels = current_vertex_labels.copy()

            # Label the current edges
            current_edge_labels = np.zeros_like(binary_image, dtype=np.int32)
            num_edges = len(self.segments)
            if num_edges > 0:
                for i, segment in enumerate(self.segments, start=1):
                    coords = segment['coords']
                    current_edge_labels[coords[:, 0], coords[:, 1]] = i
                if previous_edge_labels is not None:
                    # Compute overlap between current edges and previous edges
                    overlap = previous_edge_labels * (current_edge_labels > 0)
                    for label in range(1, num_edges + 1):
                        mask = (current_edge_labels == label)
                        overlapping_labels = np.unique(overlap[mask])
                        overlapping_labels = overlapping_labels[overlapping_labels > 0]
                        if overlapping_labels.size > 0:
                            # Assign the most frequent overlapping label
                            assigned_label = np.bincount(overlapping_labels).argmax()
                            current_edge_labels[mask] = assigned_label
                            # Update t_end in edge_times
                            edge_times.loc[edge_times['label'] == assigned_label, 't_end'] = t
                        else:
                            # Assign a new label
                            assigned_label = next_edge_label
                            current_edge_labels[mask] = assigned_label
                            edge_times = edge_times.append({'label': assigned_label, 't_start': t, 't_end': t},
                                                           ignore_index=True)
                            next_edge_label += 1
                else:
                    # First frame, assign labels as is
                    for label in range(1, num_edges + 1):
                        assigned_label = next_edge_label
                        mask = (current_edge_labels == label)
                        current_edge_labels[mask] = assigned_label
                        edge_times = edge_times.append({'label': assigned_label, 't_start': t, 't_end': t},
                                                       ignore_index=True)
                        next_edge_label += 1

                # Update edges_df
                for label in np.unique(current_edge_labels):
                    if label == 0:
                        continue
                    if not (edges_df['label'] == label).any():
                        # Get segment corresponding to this label
                        idx = label - 1  # Adjust index since labels start from 1
                        if idx < len(self.segments):
                            segment = self.segments[idx]
                            start_pos = segment['start_pos']
                            end_pos = segment['end_pos']
                            coords = segment['coords']

                            # Measure segment width
                            segment_mask = np.zeros_like(binary_image, dtype=bool)
                            segment_mask[coords[:, 0], coords[:, 1]] = True
                            # Increase the dilation size for vein_mask
                            vein_mask = morphology.binary_dilation(segment_mask, morphology.disk(7)) & binary_image
                            distance_map = ndimage.distance_transform_edt(vein_mask)
                            width_measure = self.get_segment_width(binary_image, segment, distance_map)

                            # Compute segment length
                            length = np.sum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))

                            edge_data = {
                                'label': label,
                                't_start': int(edge_times.loc[edge_times['label'] == label, 't_start']),
                                't_end': int(edge_times.loc[edge_times['label'] == label, 't_end']),
                                'start_y': start_pos[0],
                                'start_x': start_pos[1],
                                'end_y': end_pos[0],
                                'end_x': end_pos[1],
                                'length': length,
                            }
                            if width_measure:
                                edge_data.update(width_measure)
                            edges_df = edges_df.append(edge_data, ignore_index=True)
                    else:
                        # Update t_end
                        edges_df.loc[edges_df['label'] == label, 't_end'] = int(
                            edge_times.loc[edge_times['label'] == label, 't_end'])

                previous_edge_labels = current_edge_labels.copy()

            # Visualization of the network at this frame without zooming
            plt.figure(figsize=(8, 8))
            plt.imshow(binary_image, cmap='gray')
            plt.title(f"Network at frame {t}")
            plt.axis('off')
            plt.show()
            plt.close()  # Close the figure after display to free memory

        # After the loop, save the DataFrames to CSV
        vertices_df.sort_values('label', inplace=True)
        vertices_df.to_csv('vertices.csv', index=False)
        print("Vertices saved to vertices.csv")

        edges_df.sort_values('label', inplace=True)
        edges_df.to_csv('edges.csv', index=False)
        print("Edges saved to edges.csv")
