import argparse
import logging
import time
import simpleaudio as sa

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from bird_view_functions import plot_points_on_bird_eye_view, get_camera_perspective, plot_lines_between_nodes

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

mouse_pts=[]
filename = 'shortalarm.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='./video/vid_short.mp4')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)#resolution
    if w>0 and h>0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #Utils per fare la bird_view
    scale_w = 1.2 / 2
    scale_h = 4 / 2

    solid_back_color = (40, 40, 40)
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    bird_movie = cv2.VideoWriter("Pedestrian_bird.avi", fourcc, fps, (int(w * scale_w), int(h * scale_h)))

    # Initialize necessary variables
    frame_num = 0

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    num_mouse_points = 0
    first_frame_display = True

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        frame_num += 1
        ret_val, image = cap.read()

        frame_h = image.shape[0]
        frame_w = image.shape[1]

        if frame_num == 1:
            # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
            while True:
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("image")
                    break
                first_frame_display = False
            four_points = mouse_pts

            # Get perspective
            #prospettiva per bird_view
            M = get_camera_perspective(image, four_points[0:4])
            #prospettiva per altezza persona
            pts = src = np.float32(np.array([four_points[4:]]))
            warped_pt = cv2.perspectiveTransform(pts, M)[0]
            # Calcolo distanza di riferimento--> ovvero l'altezza di una persona che è circa 1.75m
            d_thresh = np.sqrt((warped_pt[0][0] - warped_pt[1][0])**2 + (warped_pt[0][1] - warped_pt[1][1])**2)
            #costruisco la bird_view
            bird_image = np.zeros((int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8)
            bird_image[:] = solid_back_color

        print("Processing frame: ", frame_num)

        # draw polygon of ROI
        pts = np.array(
            [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 255), thickness=3)

        logger.debug('image process+')
        #invdividuo le persone // Detect person
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)#add
        print('humans: ',humans)

        if not args.showBG:
            image = np.zeros(image.shape)
        logger.debug('image process+')
        image, humansFeetPoints = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        print('humanFeetPoints', humansFeetPoints)
        logger.debug('show+')#add
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.putText(image, 'WARNING', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow('tf-pose-estimation result', image)

        #disegno i punti sulla bird_view
        warped_pts, bird_image = plot_points_on_bird_eye_view(image, humansFeetPoints, M, scale_w, scale_h)
        #calcolo le linee di violazione distanza sociale, e gli passo i punti della bird view, bird image su cui scrivo, e la distanza di riferimento(circa 2m)
        social_distance_violations, good_social_distance, pairs = plot_lines_between_nodes(warped_pts, bird_image, d_thresh)
        #controllo se ci sono violazioni di distanza sociale
        if social_distance_violations > 0:
            cv2.putText(image, 'WARNING', (int(frame_w/2)-100, int(frame_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.imshow('tf-pose-estimation result', image)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until sound has finished playing

        bird_movie.write(bird_image)

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
