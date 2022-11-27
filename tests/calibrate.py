import copy
import os
import logging
import argparse
import cv2
import numpy as np
import utilities.blob_analysis as blob_analysis
import camera_distortion_calibration.radial_distortion as radial_calib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
    imageFilepath,
    outputDirectory,
    adaptiveThresholdBlockSide,
    adaptiveThresholdBias,
    correlationThreshold
):
    logging.info("calibrate.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    checkerboard_img = cv2.imread(imageFilepath)
    annotated_img = copy.deepcopy(checkerboard_img)

    checkerboard_intersections = radial_calib.CheckerboardIntersections(
        adaptive_threshold_block_side=adaptiveThresholdBlockSide,
        adaptive_threshold_bias=adaptiveThresholdBias,
        correlation_threshold=correlationThreshold,
        debug_directory=outputDirectory
    )
    intersections_list = checkerboard_intersections.FindIntersections(checkerboard_img)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilepath', help="The image filepath. Default: '../images/cam_left.png'",
                        default='../images/cam_left.png')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs_calibrate'", default='./outputs_calibrate')
    parser.add_argument('--adaptiveThresholdBlockSide', help="For adaptive threshold, the side of the neighborhood. Default: 17", type=int, default=17)
    parser.add_argument('--adaptiveThresholdBias', help="For adaptive threshold, the bias C. Default: -10", type=int, default=-10)
    parser.add_argument('--correlationThreshold', help="The correlation threshold. Default: 0.8", type=float, default=0.8)
    args = parser.parse_args()
    main(
        args.imageFilepath,
        args.outputDirectory,
        args.adaptiveThresholdBlockSide,
        args.adaptiveThresholdBias,
        args.correlationThreshold
    )