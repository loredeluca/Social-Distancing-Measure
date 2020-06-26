import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_camera_perspective(img, src_points):
    image_h = img.shape[0]
    image_w = img.shape[1]
    #coordinate dei vertici nell'immagine sorgente, ovvero i 4 punti rilevati col mouse
    src = np.float32(np.array(src_points))
    #coordinate dei vertici nell'immagine destinazione, cioè la bird_view
    dst = np.float32([[0, image_h], [image_w, image_h], [0, 0], [image_w, 0]])

    #ottengo la matrice di trasformazione prospettica
    M = cv2.getPerspectiveTransform(src, dst)
    #M_inv = cv2.getPerspectiveTransform(dst, src)

    return M#, M_inv


def plot_points_on_bird_eye_view(frame, humansFeetPoints, M, scale_w, scale_h):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    node_radius = 10
    color_node = (255, 255, 255)
    thickness_node = 5
    solid_back_color = (40, 40, 40)

    blank_image = np.zeros((int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8)
    blank_image[:] = solid_back_color
    warped_pts = []
    for i in range(len(humansFeetPoints)):
        if not isinstance(humansFeetPoints[i][1][0], str):
            mid_point_x = humansFeetPoints[i][1][0]
            mid_point_y = humansFeetPoints[i][1][1]

            pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
            warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
            warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]
            warped_pts.append(warped_pt_scaled)
            bird_image = cv2.circle(blank_image, (warped_pt_scaled[0], warped_pt_scaled[1]), node_radius, color_node, thickness_node)


    # Display Birdeye view
    #cv2.imshow("Bird Eye View", bird_image)
    #cv2.waitKey(1)
    return warped_pts, bird_image


def plot_lines_between_nodes(warped_points, bird_image, d_thresh):
    p = np.array(warped_points)
    #calcolo la distanza tra tutte le coppie di punti
    pairwise_dist = pdist(p)
    dist_matrix = squareform(pairwise_dist)

    # Good Social Distance
    #dd = np.where(dist_matrix < 10 / 6 * d_thresh)#d_thresh * 6 / 10)
    dd = np.where(dist_matrix > d_thresh * 2 / 1.75)

    close_p = []
    GREEN = (0, 255, 0) #(80, 172, 110)
    lineThickness = 3
    good_social_distance = len(np.where(pairwise_dist > d_thresh * 2 / 1.75)[0])
    print('Good social distance ', good_social_distance)

    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            close_p.append([point1, point2])

            cv2.line(bird_image, (p[point1][0], p[point1][1]), (p[point2][0], p[point2][1]), GREEN, lineThickness,)

    # Social Distance not Respected
    dd = np.where(dist_matrix <= d_thresh * 2 / 1.75)
    social_distance_violations = len(np.where(pairwise_dist <= d_thresh * 2 / 1.75)[0])
    print('Social Distance Violations ', good_social_distance)
    total_pairs = len(pairwise_dist)
    danger_p = []
    RED = (0, 0, 255)#(52, 92, 227)
    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            danger_p.append([point1, point2])
            cv2.line(bird_image, (p[point1][0], p[point1][1]), (p[point2][0], p[point2][1]), RED, lineThickness)

    # Display Birdeye view
    cv2.imshow("Bird Eye View", bird_image)
    cv2.waitKey(1)

    return social_distance_violations, good_social_distance, total_pairs
