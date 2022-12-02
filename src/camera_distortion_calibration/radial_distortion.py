import copy
import os
import logging
import argparse
import cv2
import numpy as np
import torch

class RadialDistortion():
    def __init__(self, center=None, alpha=None):
        self.center = center
        self.alpha = alpha

    def GroupCheckerboardPoints(self, intersection_points_list, grid_shapeHW):
        if len(intersection_points_list) != grid_shapeHW[0] * grid_shapeHW[1]:
            raise ValueError(f"RadialDistortion.GroupCheckerboardPoints(): len(intersection_points_list) ({len(intersection_points_list)}) != grid_shapeHW[0] * grid_shapeHW[1] ({grid_shapeHW[0] * grid_shapeHW[1]})")

        number_of_horizontal_lines = grid_shapeHW[0]
        number_of_vertical_lines = grid_shapeHW[1]

        # Vertical lines
        p_sorted_by_y = sorted(intersection_points_list, key=lambda x: x[1])

        horizontal_lines = []
        for horizontal_line_ndx in range(number_of_horizontal_lines):
            start_ndx = horizontal_line_ndx * number_of_vertical_lines
            end_ndx = (horizontal_line_ndx + 1) * number_of_vertical_lines
            horizontal_line = p_sorted_by_y[start_ndx: end_ndx]
            horizontal_lines.append(horizontal_line)

        # Vertical lines
        p_sorted_by_x = sorted(intersection_points_list, key=lambda x: x[0])

        vertical_lines = []
        for vertical_line_ndx in range(number_of_vertical_lines):
            start_ndx = vertical_line_ndx * number_of_horizontal_lines
            end_ndx = (vertical_line_ndx + 1) * number_of_horizontal_lines
            vertical_line = p_sorted_by_x[start_ndx: end_ndx]
            vertical_lines.append(vertical_line)

        return horizontal_lines, vertical_lines


class DistortionParametersOptimizer(torch.nn.Module):
    def __init__(self, center, alpha):
        super(DistortionParametersOptimizer, self).__init__()
        self.center = center
        self.alpha = alpha
        self.zero_threshold = 0.000001

    def forward(self, input_tsr):  # input_tsr.shape = (N, n_points, 2)
        shifted_points_tsr = input_tsr - self.center

    def Line(self, points_tsr):  # points_tsr.shape = (n_points, 2)
        xs = points_tsr[:, 0]
        min_x = min(xs)
        max_x = max(xs)
        if abs(min_x - max_x).item() <= self.zero_threshold:  # Vertical line
            return (min_x, torch.tensor(0))
        else:  # Non-vertical line
            ys = points_tsr[:, 1]
            min_y = min(ys)
            max_y = max(ys)
            if abs(min_y - max_y).item() <= self.zero_threshold:  # Horizontal line
                return (min_y, torch.tensor(torch.pi/2))
            else:  # Non-horizontal line
                # | x_i    y_i   -1  | | cos(theta) | = | 0 |
                # | ...    ...   ... | | sin(theta) | = | 0 |
                # | ...    ...   ... | |    rho     |   |...|
                A = torch.zeros((points_tsr.shape[0], 3))
                for row in range(points_tsr.shape[0]):
                    x_i = points_tsr[row][0]
                    y_i = points_tsr[row][1]
                    A[row, 0] = x_i
                    A[row, 1] = y_i
                    A[row, 2] = -1
                # Solution to a system of homogeneous linear equations. Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
                e_vals, e_vecs = torch.linalg.eig(torch.matmul(A.T, A))
                print(f"DistortionParametersOptimizer.Line(): torch.matmul(A.T, A) = {torch.matmul(A.T, A)}")
                # Extract the eigenvector (column) associated with the minimum eigenvalue
                print (f"DistortionParametersOptimizer.Line(): e_vals = {e_vals}")
                z = e_vecs[:, torch.argmin(e_vals.real)]
                # Multiply by a factor such that cos^2(theta) + sin^2(theta) = 1
                r2 = z[0] ** 2 + z[1] ** 2
                if abs(r2) < self.zero_threshold:
                    raise ValueError(f"DistortionParametersOptimizer.Line(): z[0]**2 + z[1]**2 ({r2}) < {self.zero_threshold}")
                z = z / torch.sqrt(r2)
                print(f"DistortionParametersOptimizer.Line(): z = {z}")
                #print
                theta = torch.angle(torch.complex(z[0].real, z[1].real))
                rho = z[2].real
                return (rho, theta)
