#!/usr/bin/env python3
"""
This script contains the class for analyzing fractals out of a binary image
"""

import numpy as np
import cv2
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import os

def linear_model(x, a, b):
    return a * x + b

class FractalAnalysis:
    def __init__(self):
        """
        Initialize FractalAnalysis class
        """
        self.binary_image = None
        self.fractal_contours = []
        self.fractal_mesh = None
        self.minkowski_dimension_cell = None
        self.minkowski_dimension_network = None

    def detect_fractal(self, binary_image, threshold=100):
        """
        Find the contours of the fractal in the binary image
        :param binary_image: 2D binary image
        :type binary_image: numpy.ndarray
        :param threshold: Area threshold to filter contours
        :type threshold: int
        """
        self.binary_image = binary_image
        self.fractal_contours = []
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > threshold:
                self.fractal_contours.append(contour)

    def extract_fractal(self):
        """
        Extract the fractal mesh from the binary image
        """
        # Create a mask to extract the fractal
        self.fractal_mesh = np.zeros(self.binary_image.shape, dtype=np.uint8)

        for contour in self.fractal_contours:
            cv2.drawContours(self.fractal_mesh, [contour], 0, 1, thickness=cv2.FILLED)

    def get_minkowski_dimension(self, fractal_mask):
        """
        Compute the Minkowski dimension of the fractal mask
        :param fractal_mask: Binary mask of the fractal
        :type fractal_mask: numpy.ndarray
        :return: Minkowski dimension
        :rtype: float
        """
        if np.any(fractal_mask):
            sizes = []
            counts = []
            # Define the range of box sizes
            max_power = int(np.log2(min(fractal_mask.shape)))
            scales = [2 ** i for i in range(1, max_power)]
            for scale in scales:
                step = scale
                total_boxes = 0
                for i in range(0, fractal_mask.shape[0], step):
                    for j in range(0, fractal_mask.shape[1], step):
                        if np.any(fractal_mask[i:i + step, j:j + step]):
                            total_boxes += 1
                counts.append(total_boxes)
            scales_log = np.log(1 / np.array(scales))
            counts_log = np.log(np.array(counts))
            coeffs, _ = curve_fit(linear_model, scales_log, counts_log)
            dimension = coeffs[0]
            return dimension
        else:
            return 0

    def fractal_descriptions(self, data):
        """
        Compute fractal descriptors over a time-lapse sequence
        :param data: 3D numpy array containing the time-lapse images (time, height, width)
        :type data: numpy.ndarray
        """
        num_frames = data.shape[0]
        frame_height, frame_width = data.shape[1], data.shape[2]

        # Initialize DataFrame to store results
        fractal_results = pd.DataFrame(columns=['frame', 'minkowski_dimension_cell', 'minkowski_dimension_network'])

        # Prepare video writer
        out = cv2.VideoWriter('fractal_analysis.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

        for t in range(num_frames):
            print(f"Processing frame {t + 1}/{num_frames}")
            # Compute cell fractal dimension (entire cell)
            binary_image_cell = (data[t] > 0).astype(np.uint8)
            self.detect_fractal(binary_image_cell)
            self.extract_fractal()
            minkowski_dimension_cell = self.get_minkowski_dimension(self.fractal_mesh)

            # Compute network fractal dimension (network only)
            binary_image_network = (data[t] == 2).astype(np.uint8)
            minkowski_dimension_network = self.get_minkowski_dimension(binary_image_network)

            # Store results
            fractal_results = fractal_results.append({
                'frame': t,
                'minkowski_dimension_cell': minkowski_dimension_cell,
                'minkowski_dimension_network': minkowski_dimension_network
            }, ignore_index=True)

            # Visualization
            visualization = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            # Overlay fractal mesh for cell
            visualization[self.fractal_mesh == 1] = (0, 255, 0)  # Green
            # Overlay network
            visualization[binary_image_network == 1] = (255, 0, 0)  # Blue
            # Add text annotations
            cv2.putText(visualization, f"Frame: {t}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(visualization, f"Cell Minkowski Dim: {minkowski_dimension_cell:.4f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(visualization, f"Network Minkowski Dim: {minkowski_dimension_network:.4f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Write frame to video
            out.write(visualization)

        # Release video writer
        out.release()

        # Save results to CSV
        fractal_results.to_csv('fractal_analysis.csv', index=False)
        print("Fractal analysis results saved to 'fractal_analysis.csv'")

    def save_fractal_mesh(self, image_save_path):
        """
        Save an image representing the fractal mesh
        :param image_save_path: Path where to save the image
        :type image_save_path: str
        """
        cv2.imwrite(image_save_path, self.fractal_mesh * 255)


if __name__ == "__main__":
    # Load  data 
    data = np.load('network2.npy')  
    fractal_analyzer = FractalAnalysis()
    fractal_analyzer.fractal_descriptions(data)