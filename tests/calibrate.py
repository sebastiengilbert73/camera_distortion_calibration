import copy
import os
import logging
import argparse
import cv2
import numpy as np
import utilities.blob_analysis as blob_analysis
import camera_distortion_calibration.checkerboard as checkerboard
import camera_distortion_calibration.radial_distortion as radial_dist

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

    checkerboard_intersections = checkerboard.CheckerboardIntersections(
        adaptive_threshold_block_side=adaptiveThresholdBlockSide,
        adaptive_threshold_bias=adaptiveThresholdBias,
        correlation_threshold=correlationThreshold,
        debug_directory=outputDirectory
    )
    intersections_list = checkerboard_intersections.FindIntersections(checkerboard_img)

    radial_distortion = radial_dist.RadialDistortion()
    horizontal_lines_points, vertical_lines_points = radial_distortion.GroupCheckerboardPoints(intersections_list, (6, 6))

    for line_ndx in range(len(horizontal_lines_points)):
        horizontal_line_points = horizontal_lines_points[line_ndx]
        color = ((177 * line_ndx)%256, (665 * line_ndx)%256, (735 * line_ndx)%256)
        for p in horizontal_line_points:
            cv2.circle(annotated_img, (round(p[0]), round(p[1])), 3, color)

    for line_ndx in range(len(vertical_lines_points)):
        vertical_line_points = vertical_lines_points[line_ndx]
        color = ((641 * line_ndx)%256, (873 * line_ndx)%256, (489 * line_ndx)%256)
        for p in vertical_line_points:
            cv2.circle(annotated_img, (round(p[0]), round(p[1])), 7, color)

    annotated_img_filepath = os.path.join(outputDirectory, "calibrate_main_annotated.png")
    cv2.imwrite(annotated_img_filepath, annotated_img)


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