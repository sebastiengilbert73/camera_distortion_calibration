import copy
import os
import logging
import argparse
import cv2
import numpy as np

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
        p_sorted_by_x = sorted(intersection_points_list, key=lambda x: x[0])
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
