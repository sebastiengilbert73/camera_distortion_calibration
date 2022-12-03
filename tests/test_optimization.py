import logging
import os
import camera_distortion_calibration.radial_distortion as radial_dist
import torch
import random
import math
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.debug("test_optimization.main()")

    neural_net = radial_dist.DistortionParametersOptimizer((320, 240), 0.00000, (640, 480))
    real_rho = 89
    real_theta = 2.06
    noise_sigma = 3.0
    points_tsr = torch.zeros(20, 2)
    for pt_ndx in range(points_tsr.shape[0]):
        beta = random.randint(-200, 200)
        x = real_rho * math.cos(real_theta) + beta * math.sin(real_theta) + random.gauss(0, noise_sigma)
        y = real_rho * math.sin(real_theta) - beta * math.cos(real_theta) + random.gauss(0, noise_sigma)
        points_tsr[pt_ndx, 0] = x
        points_tsr[pt_ndx, 1] = y
    line = neural_net.Line(points_tsr)
    logging.info(f"test_optimization.main(): line = {line}")

    horizontal_points_tsr = torch.from_numpy(np.load("../images/horizontal_intersections.npy"))  # (6, 6, 2)
    vertical_points_tsr = torch.from_numpy(np.load("../images/vertical_intersections.npy"))  # (6, 6, 2)
    line_points_tsr = torch.cat([horizontal_points_tsr, vertical_points_tsr], dim=0)  # (12, 6, 2)
    #neural_net(vertical_points_tsr)





if __name__ == '__main__':
    main()