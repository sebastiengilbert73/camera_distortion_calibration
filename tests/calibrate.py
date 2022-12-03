import ast
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
    correlationThreshold,
    gridShapeHW
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

    radial_distortion = radial_dist.RadialDistortion((checkerboard_img.shape[0], checkerboard_img.shape[1]))
    horizontal_lines_points, vertical_lines_points = radial_distortion.GroupCheckerboardPoints(intersections_list, (6, 6))
    if len(horizontal_lines_points) != gridShapeHW[0]:
        raise ValueError(f"calibrate.main(): len(horizontal_lines_points) ({len(horizontal_lines_points)}) != gridShapeHW[0] ({gridShapeHW[0]})")
    if len(vertical_lines_points) != gridShapeHW[1]:
        raise ValueError(f"calibrate.main(): len(vertical_lines_points) ({len(vertical_lines_points)}) != gridShapeHW[1] ({gridShapeHW[1]})")


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

    epoch_loss_center_alpha_list = radial_distortion.Optimize(intersections_list, gridShapeHW)
    logging.info(f"radial_distortion.center = {radial_distortion.center}; radial_distortion.alpha = {radial_distortion.alpha}")
    #print(f"epoch_loss_center_alpha_list =\n{epoch_loss_center_alpha_list}")

    # Undistort the points
    for p in intersections_list:
        undistorted_p = radial_distortion.UndistortPoint(p)
        cv2.circle(annotated_img, (round(undistorted_p[0]), round(undistorted_p[1])), 3, (255, 0, 0), thickness=-1)
    # Undistort the checkerboard image
    undistorted_checkerboard_img = radial_distortion.UndistortImage(checkerboard_img)

    # Save the points as an (N, n_points, 2) tensor
    horizontal_intersections_arr = np.zeros((len(horizontal_lines_points), len(horizontal_lines_points[0]), 2), dtype=float)
    for line_ndx in range(len(horizontal_lines_points)):
        points_list = horizontal_lines_points[line_ndx]
        for point_ndx in range(len(points_list)):
            xy = points_list[point_ndx]
            horizontal_intersections_arr[line_ndx, point_ndx, 0] = xy[0]
            horizontal_intersections_arr[line_ndx, point_ndx, 1] = xy[1]
    horizontal_intersections_arr_filepath = os.path.join(outputDirectory, "calibrate_main_horizontalIntersections.npy")
    np.save(horizontal_intersections_arr_filepath, horizontal_intersections_arr)

    vertical_intersections_arr = np.zeros((len(vertical_lines_points), len(vertical_lines_points[0]), 2),
                                            dtype=float)
    for line_ndx in range(len(vertical_lines_points)):
        points_list = vertical_lines_points[line_ndx]
        for point_ndx in range(len(points_list)):
            xy = points_list[point_ndx]
            vertical_intersections_arr[line_ndx, point_ndx, 0] = xy[0]
            vertical_intersections_arr[line_ndx, point_ndx, 1] = xy[1]
    vertical_intersections_arr_filepath = os.path.join(outputDirectory, "calibrate_main_verticalIntersections.npy")
    np.save(vertical_intersections_arr_filepath, vertical_intersections_arr)

    annotated_img_filepath = os.path.join(outputDirectory, "calibrate_main_annotated.png")
    cv2.imwrite(annotated_img_filepath, annotated_img)

    undistorted_checkerboard_img_filepath = os.path.join(outputDirectory, "calibrate_main_undistortedCheckerboard.png")
    cv2.imwrite(undistorted_checkerboard_img_filepath, undistorted_checkerboard_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilepath', help="The image filepath. Default: '../images/cam_left.png'",
                        default='../images/cam_left.png')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs_calibrate'", default='./outputs_calibrate')
    parser.add_argument('--adaptiveThresholdBlockSide', help="For adaptive threshold, the side of the neighborhood. Default: 17", type=int, default=17)
    parser.add_argument('--adaptiveThresholdBias', help="For adaptive threshold, the bias C. Default: -10", type=int, default=-10)
    parser.add_argument('--correlationThreshold', help="The correlation threshold. Default: 0.8", type=float, default=0.8)
    parser.add_argument('--gridShapeHW', help="The shape of the intersections grid (height, width). Default: '(6, 6)'", default='(6, 6)')
    args = parser.parse_args()

    gridShapeHW = ast.literal_eval(args.gridShapeHW)
    main(
        args.imageFilepath,
        args.outputDirectory,
        args.adaptiveThresholdBlockSide,
        args.adaptiveThresholdBias,
        args.correlationThreshold,
        gridShapeHW
    )