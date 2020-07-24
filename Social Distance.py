# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import time
import simpleaudio as sa

from sys import platform
import argparse

from bird_view_functions import plot_points_on_bird_eye_view, get_camera_perspective, order_points, plot_distance_circle_nodes, plot_lines_between_person
import numpy as np
from utils import *

fps_time = 0
MAX_DIM_BIRD_VIEW = 5 #meters
REFERENCE_DISTANCE = 0.4
#chessboard dimension
PTS_ROW = 7
PTS_COL = 10

SHOW_IMAGE_WITH_ROI = True
mouse_pts=[]
filename = 'shortalarm.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)


def get_mouse_points(event, x, y, a, b):
    # Used to mark 4 points on the frame one of the video about the reference
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

except Exception as e:
    print(e)
    sys.exit(-1)


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='d4.mp4')
    
    args = parser.parse_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models"


    cap = cv2.VideoCapture(args.video) #start from video passed by argument
    #cap = cv2.VideoCapture(0) #start from webcam
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #initialize useful variables 
    frame_num = 0
    sliding_windows = {}
    past_people = []

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened() :
        ret_val, image = cap.read()

        if not ret_val:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_num += 1
        frame_h = image.shape[0]
        frame_w = image.shape[1]

        if frame_num == 1:

            ret1, corners = cv2.findChessboardCorners(image, (PTS_ROW, PTS_COL),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK)
            points = []
            if not ret1:
                #if chessboard is not recognited choose manually points
                cv2.namedWindow("Choose the corners of the reference (chessboard or simple paper) in order (tl,tr,br,bl)", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("Choose the corners of the reference (chessboard or simple paper) in order (tl,tr,br,bl)", get_mouse_points)
                num_mouse_points = 0
                first_frame_display = True
                # Ask user to mark four points from reference in the image in order tl, tr, br, bl.
                while True:
                    cv2.imshow("Choose the corners of the reference (chessboard or simple paper) in order (tl,tr,br,bl)", image)
                    cv2.waitKey(1)
                    if len(mouse_pts) == 5:
                        cv2.destroyWindow("Choose the corners of the reference (chessboard or simple paper) in order (tl,tr,br,bl)")
                        break
                    first_frame_display = False

                four_points = np.array(mouse_pts[0:4], dtype="float32")

                '''#find long side reference object size
                ref_pixel_w,_,long_side = find_max_side(four_points)

                #source points for reference distance
                pts = src = np.array([[long_side[0], long_side[1]]], dtype="float32")'''

            else:
                #take points from chessboard corners
                for i in range(len(corners)):
                    if (i == 0 or i == PTS_ROW - 1 or i == (PTS_ROW * PTS_COL) - PTS_ROW or i == (PTS_ROW * PTS_COL) - 1):
                        cv2.circle(image, (corners[i][0][0], corners[i][0][1]), 5, (0, 255, 255), -1)
                        points.append((corners[i][0][0], corners[i][0][1]))

                four_points = np.array(points, dtype="float32")
                
                #order points in order tl, tr, br, bl
                four_points = order_points(four_points)

            #find long side reference object size
            ref_pixel_w,_,long_side = find_max_side(four_points)

            #source points for reference distance
            pts = src = np.array([[long_side[0],long_side[1]]], dtype="float32")

            chess_center = find_center(four_points)
            cv2.circle(image, (int(chess_center[0]), int(chess_center[1])), 5, RED, -1)

            desired_distance = (MAX_DIM_BIRD_VIEW * ref_pixel_w) / REFERENCE_DISTANCE

            #find ROI limits wrt max dimension of bird view (in meters)
            crop = find_ROI_limits(image,chess_center,desired_distance)

            #compute chessboard (or reference object) angle and choose the method to find the ROI
            if math.fabs(angleBetween(four_points[3], four_points[2])) < 10 or math.fabs(angleBetween(four_points[3], four_points[2])) > 170:
                bird_points, new_left, new_right, new_up, new_down = find_roi_straight(image, four_points, crop)
            else:
                bird_points, new_left, new_right, new_up, new_down = find_roi_askew(image, four_points, crop)


            #if SHOW_IMAGE_WITH_ROI is true it shows the computed ROI on the original image
            if SHOW_IMAGE_WITH_ROI:
                image_roi = cv2.copyMakeBorder(image, int(new_up), int(new_down), int(new_left), int(new_right), cv2.BORDER_CONSTANT, BLACK)

                new_h = image_roi.shape[0]
                new_w = image_roi.shape[1]

                # draw polygon of ROI
                roi_points = np.array([bird_points[0], bird_points[1], bird_points[2], bird_points[3]], np.int32)
                cv2.polylines(image_roi, [roi_points], True, YELLOW, thickness=2)

                cv2.namedWindow('Image with Region Of Interest', cv2.WINDOW_NORMAL)
                cv2.imshow('Image with Region Of Interest', image_roi)


            # Utils per fare la bird_view
            #scale_w = 8
            #scale_h = 8
            #scale_w = bird_view_width*2/frame_w
            #scale_h = bird_view_height/frame_h

            #scale_w, scale_h = find_scale(four_points,bird_points)


            # Get perspective for bird view
            M, M_inv,scale_w, scale_h = get_camera_perspective(image, np.float32(np.array(bird_points)))

            print("SCALE_W. SCALE_H: ", scale_w, scale_h)

            #perspective for reference measure
            warped_pt = cv2.perspectiveTransform(pts, M)[0]

            # compute reference distance
            d_thresh = np.sqrt((warped_pt[0][0] - warped_pt[1][0])**2 + (warped_pt[0][1] - warped_pt[1][1])**2)

        print("Processing frame: ", frame_num)

        #initialize Datum and get human body part estimation
        datum = op.Datum()
        datum.cvInputData = image
        opWrapper.emplaceAndPop([datum])
        image = datum.cvOutputData
        keyPointsHuman = datum.poseKeypoints
        humansFeetPoints = getFeetPoints(keyPointsHuman)

        #inizialize the past people vector (FORSE SUPERFLUO)
        if frame_num == 1:
            for i in range(len(humansFeetPoints)):
                past_people.append(humansFeetPoints[i])

        #put fps number on the image
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.namedWindow('Frames with openpose body parts estimation', cv2.WINDOW_NORMAL)
        cv2.imshow('Frames with openpose body parts estimation', image)


        if len(humansFeetPoints) != 0:
            # write points on the bird view
            warped_pts, bird_image = plot_points_on_bird_eye_view(image, humansFeetPoints, M, scale_w, scale_h,d_thresh,REFERENCE_DISTANCE, bird_points, new_left,new_up,sliding_windows,past_people)
            if len(warped_pts) != 0:
                #draw distancing lines between people in the normal view
                plot_lines_between_person(image, warped_pts, d_thresh, REFERENCE_DISTANCE, scale_w, scale_h, M_inv, new_left, new_up)
                #draw distancing circles between people in the bird view
                social_distance_violations, good_social_distance, pairs = plot_distance_circle_nodes(warped_pts, bird_image, d_thresh, REFERENCE_DISTANCE)
                #check for social distance violations 
                if social_distance_violations > 0:
                    cv2.putText(image, 'WARNING', (int(frame_w / 2) - 100, int(frame_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 4)
                    cv2.imshow('Frames with openpose body parts estimation', image)
                    #audio warning alarm
                    #play_obj = wave_obj.play()
                    #play_obj.wait_done()


            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        cv2.imshow('Frames with openpose body parts estimation', image)
        cv2.waitKey(1)
        cv2.namedWindow('Bird Eye View', cv2.WINDOW_NORMAL)
        cv2.imshow("Bird Eye View", bird_image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()