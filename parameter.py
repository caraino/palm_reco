import numpy as np
import cv2
import math

imageWidth = 720
imageHeight = 576
imageSize = (imageWidth, imageHeight)
global PalmDetect
PalmDetect = cv2.CascadeClassifier('palm_cascade.xml')

'''leftdate
fc_left_x   0            cc_left_x
0           fc_left_y    cc_left_y
0           0            1
'''
cameraMatrixL = np.array([[512.69910, 0, 269.92130],
                          [0, 511.20007, 232.16848],
                          [0, 0, 1]])
# [kc_left_01,  kc_left_02,  kc_left_03,  kc_left_04,   kc_left_05]
distCoeffL = np.array([0.00076, 0.00074, -0.00079, -0.00035, 0.00000])

'''rightdate
fc_right_x   0              cc_right_x
0            fc_right_y     cc_right_y
0            0              1
'''
cameraMatrixR = np.array([[512.40724, 0, 316.19848],
                          [0, 510.89134, 222.78870],
                          [0, 0, 1]])

# kc_right_01,  kc_right_02,  kc_right_03,  kc_right_04,   kc_right_05
distCoeffR = np.array([0.00427, -0.00618, -0.00068, -0.00155, 0.00000])

T = np.array([-26.58342, -1.11175, 1.29023])

rec = np.array([-0.00012, 0.00450, 0.00013])

R = cv2.Rodrigues(rec)[0]
Rl, Rr, Pl, Pr, Q, validROIL, validROIR = cv2.stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR,
                                                            imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
                                                            newImageSize=imageSize)
global mapLx, mapLy,mapRx, mapRy

mapLx, mapLy = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, cv2.CV_32FC1)

global cap2
cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

global cap1
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)