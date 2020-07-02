import cv2
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

def get_camera_perspective(img, src_points):
    image_h = img.shape[0]
    image_w = img.shape[1]
    #coordinate dei vertici nell'immagine sorgente, ovvero i 4 punti rilevati col mouse
    src = np.float32(np.array(src_points))
    #coordinate dei vertici nell'immagine destinazione, cioè la bird_view
    dst = np.float32([[0, image_h], [image_w, image_h], [0, 0], [image_w, 0]])

    #ottengo la matrice di trasformazione prospettica
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv


def plot_points_on_bird_eye_view(frame, humansFeetPoints, M, scale_w, scale_h, d_thresh, ref_dist):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    thickness_node = 2
    solid_back_color = (40, 40, 40)

    blank_image = np.zeros((int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8)
    blank_image[:] = solid_back_color
    warped_pts = []
    if len(humansFeetPoints) == 0:
        bird_image = blank_image
    else:
        for i in range(len(humansFeetPoints)):
            if humansFeetPoints[i][1][0] != -1 and humansFeetPoints[i][1][1] != -1:
            #if not isinstance(humansFeetPoints[i][1][0], str):
                mid_point_x = humansFeetPoints[i][1][0]
                mid_point_y = humansFeetPoints[i][1][1]

                pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
                warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
                warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]
                warped_pts.append(warped_pt_scaled)
                bird_image = cv2.circle(blank_image, (warped_pt_scaled[0], warped_pt_scaled[1]), 5, WHITE, thickness_node)

                _, pairwise_dist, _ = find_pairwise_distance(warped_pts)

                for dist in pairwise_dist:
                    if dist > d_thresh * 2 / ref_dist:
                        bird_image = cv2.circle(bird_image, (warped_pt_scaled[0], warped_pt_scaled[1]),
                                            int((d_thresh * 2 / ref_dist) / 2), GREEN, 1)
                    elif dist <= d_thresh * 2 / ref_dist:
                        bird_image = cv2.circle(bird_image, (warped_pt_scaled[0], warped_pt_scaled[1]),
                                            int((d_thresh * 2 / ref_dist) / 2), RED, 1)
            else:
                warped_pts.append([-1, -1])
                bird_image = blank_image

    # Display Birdeye view
    # cv2.imshow("Bird Eye View", bird_image)
    # cv2.waitKey(1)
    return warped_pts, bird_image


def plot_lines_between_person(image, warped_points, d_thresh, ref_dist, scale_w, scale_h, Minv):
    p, pairwise_dist, dist_matrix = find_pairwise_distance(warped_points)

    # Good Social Distance
    dist = np.where(dist_matrix > d_thresh * 2 / ref_dist)
    for i in range(int(np.ceil(len(dist[0]) / 2))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            point2 = dist[1][i]

            # trasformo le coordinate dei pts trovati per la bird_view in coordinate per la normal view
            w_pt1 = np.array([[[int(p[point1][0] / scale_w), int(p[point1][1] / scale_h)]]], dtype="float32")
            w_pt2 = np.array([[[int(p[point2][0] / scale_w), int(p[point2][1] / scale_h)]]], dtype="float32")
            warped_pt1 = cv2.perspectiveTransform(w_pt1, Minv)[0][0]
            warped_pt2 = cv2.perspectiveTransform(w_pt2, Minv)[0][0]

            cv2.line(image, (warped_pt1[0], warped_pt1[1]), (warped_pt2[0], warped_pt2[1]), GREEN, 1)

            mid_x = (warped_pt1[0] + warped_pt2[0])/2
            mid_y = (warped_pt1[1] + warped_pt2[1]) / 2
            d = int(math.sqrt((warped_pt1[0]-warped_pt2[0])**2+(warped_pt1[1]-warped_pt2[1])**2))
            x = str(round(((d * ref_dist) / d_thresh), 2))+' m'
            cv2.putText(image, x, (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    # Social Distance not Respected
    dist = np.where(dist_matrix <= d_thresh * 2 / ref_dist)
    for i in range(int(np.ceil(len(dist[0]) / 2))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            point2 = dist[1][i]

            # trasformo le coordinate dei pts trovati per la bird_view in coordinate per la normal view
            w_pt1 = np.array([[[int(p[point1][0] / scale_w), int(p[point1][1] / scale_h)]]], dtype="float32")
            w_pt2 = np.array([[[int(p[point2][0] / scale_w), int(p[point2][1] / scale_h)]]], dtype="float32")
            warped_pt1 = cv2.perspectiveTransform(w_pt1, Minv)[0][0]
            warped_pt2 = cv2.perspectiveTransform(w_pt2, Minv)[0][0]

            cv2.line(image, (warped_pt1[0], warped_pt1[1]), (warped_pt2[0], warped_pt2[1]), RED, 1)

            mid_x = (warped_pt1[0] + warped_pt2[0]) / 2
            mid_y = (warped_pt1[1] + warped_pt2[1]) / 2
            d = int(math.sqrt((warped_pt1[0] - warped_pt2[0]) ** 2 + (warped_pt1[1] - warped_pt2[1]) ** 2))
            x = str(round(((d * ref_dist) / d_thresh), 2)) + ' m'
            cv2.putText(image, x, (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    #cv2.imshow('tf-pose-estimation result', image) # questi ci sono
    #cv2.waitKey(1)


def plot_distance_circle_nodes(warped_points, bird_image, d_thresh, ref_dist):
    p, pairwise_dist, dist_matrix = find_pairwise_distance(warped_points)

    # Good Social Distance
    dist = np.where(dist_matrix > d_thresh * 2 / ref_dist)

    good_social_distance = len(np.where(pairwise_dist > d_thresh * 2 / ref_dist)[0])
    print('Good social distance ', good_social_distance)

    for i in range(int(np.ceil(len(dist[0]) / 2))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            #point2 = dist[1][i]

            cv2.circle(bird_image, (p[point1][0], p[point1][1]), int((d_thresh * 2 / ref_dist)/2), GREEN, 1)

    # Social Distance not Respected
    dist = np.where(dist_matrix <= d_thresh * 2 / ref_dist)
    social_distance_violations = len(np.where(pairwise_dist <= d_thresh * 2 / ref_dist)[0])
    print('Social Distance Violations ', social_distance_violations)
    total_pairs = len(pairwise_dist)

    for i in range(int(np.ceil(len(dist[0]) / 2))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            point2 = dist[1][i]

            cv2.circle(bird_image, (p[point1][0], p[point1][1]), int((d_thresh * 2 / ref_dist) / 2), RED, 1)
            cv2.circle(bird_image, (p[point2][0], p[point2][1]), int((d_thresh * 2 / ref_dist) / 2), RED, 1)

    # Display Birdeye view
    #cv2.namedWindow('Bird Eye View', cv2.WINDOW_NORMAL) #ci sono
    #cv2.imshow("Bird Eye View", bird_image)
    #cv2.waitKey(1)

    return social_distance_violations, good_social_distance, total_pairs


def find_pairwise_distance(warped_points):
    p = np.array(warped_points)
    print('p', p)
    # calcolo la distanza tra tutte le coppie di punti
    pairwise_dist = pdist(p)
    # matrice delle distanze
    dist_matrix = squareform(pairwise_dist)

    return p, pairwise_dist, dist_matrix
