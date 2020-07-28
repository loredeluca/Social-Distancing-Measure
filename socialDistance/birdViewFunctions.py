import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from utils import compare_vect, compare_and_get_index, find_max_side
from collections import deque

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
SOLID_BACK_COLOR = (40, 40, 40)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered in the order (tl,tr,br,bl)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_max_dimension(pts):
    #return max dimesions(width,height) given four rectangle points 
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    return maxWidth,maxHeight

def get_camera_perspective(img, src_points):
    # method that compute the trasformation matrix given source points.

    bird_w, bird_h, _, _ = find_max_side(src_points)

    # coordinates of the vertices in the source image, or the 4 points detected
    # with the mouse or automatically by the chessboard
    src = np.float32(np.array(src_points))

    #coordinates of the vertices in the target image, i.e. the bird_view
    dst = np.float32([[0, 0], [bird_w, 0], [bird_w, bird_h], [0, bird_h]])

    #get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv


def find_mid_points(sliding_window):
    #return the mid points (X and Y) in the sliding window or -1,-1 if empty.
    if len(sliding_window) != 0:
        mid_X = 0
        mid_Y = 0
        index = 0
        for points in sliding_window:
            index = index + 1
            mid_X = mid_X + points[0]
            mid_Y = mid_Y + points[1]
        mid_X = mid_X / index
        mid_Y = mid_Y / index
        return mid_X,mid_Y
    else:
        return -1,-1


def find_mid_points_body(body_parts):
    #returns the mid points (X and Y) in the body parts list or -1,-1 if empty.
    if len(body_parts) != 0:
        mid_X = 0
        mid_Y = 0
        index = 0
        for i in range(len(body_parts)):

            if (body_parts[i][0] != 0.0 and body_parts[i][1] != 0.0):
                index = index + 1
                mid_X = mid_X + body_parts[i][0]
                mid_Y = mid_Y + body_parts[i][1]
        if index == 0:
            mid_X = -1
            mid_Y = -1
        else:
            mid_X = mid_X / index
            mid_Y = mid_Y / index
        return mid_X, mid_Y
    else:
        return -1, -1


def check_humans_positions(humans,past_people):
    '''check the humans in the image. 
    if a person is new in the scene it will be added as a new entry person
    if a person is not new it will be update his position and get his index in the temporal order in the video.'''

    if len(humans) == 0:
        return None,None
    people = {}

    for i in range(len(humans)):
        found = False
        mid_x_h, mid_y_h = find_mid_points_body(humans[i])
        if (mid_x_h != -1 and mid_y_h != -1):
            for j in range(len(past_people)):

                mid_x_p, mid_y_p = find_mid_points_body(past_people[j])
                if (mid_x_p != -1 and mid_y_p != -1):

                    if (math.fabs(mid_x_h - mid_x_p) < 80 and math.fabs(mid_y_h - mid_y_p) < 80):
                        found = True
                        past_people[j] = humans[i]
                        people[j] = humans[i]
                        break
            if not found:
                past_people.append(humans[i])

                index = compare_and_get_index(past_people,humans[i])

                people[index] = humans[i]
    return people

def plot_points_on_bird_eye_view(humansFeetPoints, M, scale_w, scale_h,d_thresh, ref_dist, bird_points, left , up, sliding_windows,past_people):
    #the method draw warped people coordinates on a white image that will be our bird view (the past computed ROI)

    node_radius = 20
    thickness_node = -1
    bird_points = np.array([bird_points], dtype="float32")
    warped_bird_points = cv2.perspectiveTransform(bird_points, M)
    bird_width, bird_height = get_max_dimension(warped_bird_points[0])
    bird_image = np.zeros((int(bird_height*scale_h), int(bird_width*scale_w), 3), np.uint8)
    bird_image[:] = SOLID_BACK_COLOR
    warped_pts = []
    if len(humansFeetPoints.shape) != 0:

        people = check_humans_positions(humansFeetPoints,past_people)
        list_people = [(v) for k, v in people.items()]
        humansFeetPoints = np.array(list_people,dtype="float64")

        for i in range(len(humansFeetPoints)):
            if (humansFeetPoints[i][0][0] != -1 and humansFeetPoints[i][0][1] != -1) or (humansFeetPoints[i][1][0] != -1 and humansFeetPoints[i][1][1] != -1):
                mid_point_x_dx = humansFeetPoints[i][0][0] + left
                mid_point_y_dx = humansFeetPoints[i][0][1] + up
                mid_point_x_sx = humansFeetPoints[i][1][0] + left
                mid_point_y_sx = humansFeetPoints[i][1][1] + up

                for k, v in people.items():
                    if compare_vect(v,humansFeetPoints[i]):
                        target_index = k

                if target_index in sliding_windows.keys():
                    if len(sliding_windows[target_index][0]) == 4:
                        sliding_windows[target_index][0].popleft()
                    if len(sliding_windows[target_index][1]) == 4:
                        sliding_windows[target_index][1].popleft()
                else:
                    sliding_windows[target_index] = [deque(),deque()]

                sliding_windows[target_index][0].append((mid_point_x_sx, mid_point_y_sx))
                sliding_windows[target_index][1].append((mid_point_x_dx, mid_point_y_dx))

                mid_X_sx, mid_Y_sx = find_mid_points(sliding_windows[target_index][0])
                mid_X_dx, mid_Y_dx = find_mid_points(sliding_windows[target_index][1])

                final_mid_X = (mid_X_dx + mid_X_sx) / 2
                final_mid_Y = (mid_Y_dx + mid_Y_sx) / 2

                polygon = Polygon([bird_points[0][0], bird_points[0][1], bird_points[0][2], bird_points[0][3]])

                point = Point(final_mid_X, final_mid_Y)
                
                if polygon.contains(point):

                    pts = np.array([[[final_mid_X, final_mid_Y]]], dtype="float32")
                    #actual_pts = np.array([[[actual_mid_X, actual_mid_Y]]], dtype="float32")

                    warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
                    #actual_warped_pt = cv2.perspectiveTransform(actual_pts, M)[0][0]

                    warped_pt_scaled = [int(warped_pt[0]*scale_w), int(warped_pt[1]*scale_h)]
                    #actual_warped_pt_scaled = [int(actual_warped_pt[0] * scale_w), int(actual_warped_pt[1] * scale_h)]

                    warped_pts.append(warped_pt_scaled)
                    bird_image = cv2.circle(bird_image, (warped_pt_scaled[0], warped_pt_scaled[1]), node_radius, WHITE, thickness_node)
                    _, pairwise_dist, _ = find_pairwise_distance(warped_pts)

                    for dist in pairwise_dist:
                        if dist > d_thresh * 2 / ref_dist:

                            bird_image = cv2.circle(bird_image, (warped_pt_scaled[0], warped_pt_scaled[1]),int((d_thresh * 2 / ref_dist) / 2), GREEN, 2)

                        elif dist <= d_thresh * 2 / ref_dist:

                            bird_image = cv2.circle(bird_image, (warped_pt_scaled[0], warped_pt_scaled[1]),int((d_thresh * 2 / ref_dist) / 2), RED, 2)

    return warped_pts, bird_image


def plot_lines_between_person(image, warped_points, d_thresh, ref_dist, scale_w, scale_h, Minv, sx, up):
    #this method draw lines between people in the ROI and write on it the distance in meters 

    p, pairwise_dist, dist_matrix = find_pairwise_distance(warped_points)
    # Good Social Distance
    dist = np.where(dist_matrix > d_thresh * 2 / ref_dist)
    for i in range(int(np.ceil(len(dist[0]) ))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            point2 = dist[1][i]

            if p[point1][0] != -1 and p[point1][1] != -1 and p[point2][0] != -1 and p[point2][1] != -1:

                # transform the coordinates of the points found for the bird_view into coordinates for the normal view
                w_pt1 = np.array([[[int(p[point1][0] / scale_w), int(p[point1][1] / scale_h)]]], dtype="float32")
                w_pt2 = np.array([[[int(p[point2][0] / scale_w), int(p[point2][1] / scale_h)]]], dtype="float32")
                warped_pt1 = cv2.perspectiveTransform(w_pt1, Minv)[0][0]
                warped_pt2 = cv2.perspectiveTransform(w_pt2, Minv)[0][0]

                #center the coordinates in (0,0)
                warped_pt1[0] = warped_pt1[0] - sx
                warped_pt1[1] = warped_pt1[1] - up
                warped_pt2[0] = warped_pt2[0] - sx
                warped_pt2[1] = warped_pt2[1] - up

                cv2.line(image, (warped_pt1[0], warped_pt1[1]), (warped_pt2[0], warped_pt2[1]), GREEN, 1)

                mid_x = (warped_pt1[0] + warped_pt2[0])/2
                mid_y = (warped_pt1[1] + warped_pt2[1]) / 2
                d = int(math.sqrt((warped_pt1[0]-warped_pt2[0])**2+(warped_pt1[1]-warped_pt2[1])**2))
                x = str(round(((d * ref_dist) / d_thresh), 2))+' m'
                cv2.putText(image, x, (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    # Social Distance not Respected
    dist = np.where(dist_matrix <= d_thresh * 2 / ref_dist)
    for i in range(int(np.ceil(len(dist[0]) ))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            point2 = dist[1][i]

            if p[point1][0] != -1 and p[point1][1] != -1 and p[point2][0] != -1 and p[point2][1] != -1:

                # transform the coordinates of the points found for the bird_view into coordinates for the normal view
                w_pt1 = np.array([[[int(p[point1][0] / scale_w), int(p[point1][1] / scale_h)]]], dtype="float32")
                w_pt2 = np.array([[[int(p[point2][0] / scale_w), int(p[point2][1] / scale_h)]]], dtype="float32")
                warped_pt1 = cv2.perspectiveTransform(w_pt1, Minv)[0][0]
                warped_pt2 = cv2.perspectiveTransform(w_pt2, Minv)[0][0]

                # center the coordinates in (0,0)
                warped_pt1[0] = warped_pt1[0] - sx
                warped_pt1[1] = warped_pt1[1] - up
                warped_pt2[0] = warped_pt2[0] - sx
                warped_pt2[1] = warped_pt2[1] - up

                cv2.line(image, (warped_pt1[0], warped_pt1[1]), (warped_pt2[0], warped_pt2[1]), RED, 1)

                mid_x = (warped_pt1[0] + warped_pt2[0]) / 2
                mid_y = (warped_pt1[1] + warped_pt2[1]) / 2
                d = int(math.sqrt((warped_pt1[0] - warped_pt2[0]) ** 2 + (warped_pt1[1] - warped_pt2[1]) ** 2))
                x = str(round(((d * ref_dist) / d_thresh), 2)) + ' m'
                cv2.putText(image, x, (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)


def plot_distance_circle_nodes(warped_points, bird_image, d_thresh, ref_dist):
    #this method draw circle around people based on a computed distance measure

    p, pairwise_dist, dist_matrix = find_pairwise_distance(warped_points)

    # Good Social Distance
    dist = np.where(dist_matrix > d_thresh * 2 / ref_dist)

    good_social_distance = len(np.where(pairwise_dist > d_thresh * 2 / ref_dist)[0])
    print('Good social distance ', good_social_distance)

    for i in range(int(np.ceil(len(dist[0]) / 2))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]

            cv2.circle(bird_image, (p[point1][0], p[point1][1]), int((d_thresh * 2/ ref_dist) / 2), GREEN, 2)

    # Social Distance not Respected
    dist = np.where(dist_matrix <= d_thresh * 2 / ref_dist)
    social_distance_violations = len(np.where(pairwise_dist <= d_thresh * 2 / ref_dist)[0])
    print('Social Distance Violations ', social_distance_violations)
    total_pairs = len(pairwise_dist)

    for i in range(int(np.ceil(len(dist[0]) / 2))):
        if dist[0][i] != dist[1][i]:
            point1 = dist[0][i]
            point2 = dist[1][i]
            cv2.circle(bird_image, (p[point1][0], p[point1][1]), int((d_thresh * 2 / ref_dist) / 2), RED, 2)
            cv2.circle(bird_image, (p[point2][0], p[point2][1]), int((d_thresh * 2 / ref_dist) / 2), RED, 2)

    return social_distance_violations, good_social_distance, total_pairs


def find_pairwise_distance(warped_points):
    #find distance between all warped points pairs 

    p = np.array(warped_points)
    pairwise_dist = dist.pdist(p)
    # distance matrix
    dist_matrix = dist.squareform(pairwise_dist)

    return p, pairwise_dist, dist_matrix




