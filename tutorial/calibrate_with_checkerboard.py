import ast
import copy
import os
import logging
import argparse
import cv2
import numpy as np
import camera_distortion_calibration.checkerboard as checkerboard
import camera_distortion_calibration.radial_distortion as radial_dist

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.debug("calibrate_with_checkerboard.main()")

    output_directory = "./output"
    checkerboard_image_filepath = "../images/cam_left.png"
    adaptive_threshold_block_side = 17
    adaptive_threshold_bias = -10
    correlation_threshold = 0.8
    grid_shapeHW = (6, 6)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the checkerboard image
    checkerboard_img = cv2.imread(checkerboard_image_filepath)
    annotated_img = copy.deepcopy(checkerboard_img)
    # Display the checkerboard
    cv2.imshow("Checkerboard", checkerboard_img)
    cv2.waitKey(0)

    # Find the checkerboard intersections, which will be our feature points that belong to a plane
    checkerboard_intersections = checkerboard.CheckerboardIntersections(
        adaptive_threshold_block_side=adaptive_threshold_block_side,
        adaptive_threshold_bias=adaptive_threshold_bias,
        correlation_threshold=correlation_threshold,
        debug_directory=output_directory
    )
    intersections_list = checkerboard_intersections.FindIntersections(checkerboard_img)

    # Create a RadialDistortion object, that will optimize its parameters
    radial_distortion = radial_dist.RadialDistortion((checkerboard_img.shape[0], checkerboard_img.shape[1]))
    # Start the optimization
    epoch_loss_center_alpha_list = radial_distortion.Optimize(intersections_list, grid_shapeHW)


if __name__ == '__main__':
    main()