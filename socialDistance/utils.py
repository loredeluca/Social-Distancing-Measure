import math
import numpy as np
import cv2

BLACK = (0,0,0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0,255,0)

def find_ROI_limits(image,chess_center,desired_distance):
    #find the ROI given the chess center (or reference object center) and the desired distance in all directions (up,right,down,left) in pixel

    frame_h = image.shape[0]
    frame_w = image.shape[1]

    limit_points = []
    crop = []

    #find the four limit points starting from the chess center moving for desired distance in each direction.
    limit_points.append([chess_center[0] - desired_distance, chess_center[1]])
    limit_points.append([chess_center[0], chess_center[1] - desired_distance])
    limit_points.append([chess_center[0] + desired_distance, chess_center[1]])
    limit_points.append([chess_center[0], chess_center[1] + desired_distance])

    #find intersection lines to get the up left point
    L1 = line([limit_points[0][0], limit_points[0][1] - desired_distance],
              [limit_points[0][0], limit_points[0][1] + desired_distance])
    L2 = line([limit_points[1][0] - desired_distance, limit_points[1][1]],
              [limit_points[1][0] + desired_distance, limit_points[1][1]])

    #check if this point is outsize the original image size
    up_left_X, up_left_Y = intersection(L1, L2)
    if up_left_X < 0:
        up_left_X = 0
    if up_left_Y < 0:
        up_left_Y = 0
    crop.append((int(up_left_X), int(up_left_Y)))

    #find intersection lines to get the down right point
    L1 = line([limit_points[2][0], limit_points[2][1] - desired_distance],
              [limit_points[2][0], limit_points[2][1] + desired_distance])
    L2 = line([limit_points[3][0] - desired_distance, limit_points[3][1]],
              [limit_points[3][0] + desired_distance, limit_points[3][1]])

    #check if this point is outsize the original image size
    down_right_X, down_right_Y = intersection(L1, L2)
    if down_right_X > frame_w:
        down_right_X = frame_w
    if down_right_Y > frame_h:
        down_right_Y = frame_h
    crop.append((int(down_right_X), int(down_right_Y)))

    return np.array(crop, dtype="int32")

def find_center(four_points):
    #find the chessboard center using homography to warp the reference points before find the center
    #after this point will be warped in the original image space

    chess_w, chess_h, _ ,_= find_max_side(four_points)

    source = np.float32(np.array(four_points))
    dest = np.float32([[0, 0], [chess_w, 0], [chess_w, chess_h], [0, chess_h]])

    # get the perspective matrix
    M = cv2.getPerspectiveTransform(source, dest)
    M_inv = cv2.getPerspectiveTransform(dest, source)

    bird_points = np.array([four_points], dtype="float32")
    warped_points = cv2.perspectiveTransform(bird_points, M)
    warped_chess_center = warped_points[0][2] - [chess_w / 2, chess_h / 2]

    #turn the found center in a original space point
    chess_center = cv2.perspectiveTransform(np.array([[warped_chess_center]], dtype="float32"), M_inv)[0][0]

    return chess_center

def compare_and_get_index(v1,v2):
    #compare the elements of a vector of vectors(v1) and a vector(v2) and check if they are equal
    for i in range(len(v1)):
        if compare_vect(v1[i],v2):
            return i

def compare_vect(v1,v2):
    #compare the elements of two vectors and check if they are equal 
    for i in range(len(v1)):
        for j in range(len(v2)):
            if(i == j):
                for k in range(len(v1[i])):
                    for l in range(len(v2[j])):
                        if k == l:
                            if v1[i][k] != v2[j][l]:
                                return False
    return True

def find_max_side(four_points):
    #find the longer side of the chessboard and compute its dimension.
    #the method returns points of this longer side.

    #find the reference chessboard long side and the dimensions (width, height)
    left_side = find_distance(four_points[0], four_points[3])
    right_side = find_distance(four_points[1], four_points[2])
    chess_h = max(right_side, left_side)
    if chess_h == right_side:
        long_side_points = (four_points[1], four_points[2])
    else:
        long_side_points = (four_points[0], four_points[3])
    up_side = find_distance(four_points[0], four_points[1])
    down_side = find_distance(four_points[3], four_points[2])
    chess_w = max(up_side, down_side)
    if chess_w == up_side:
        short_side_points = (four_points[0], four_points[1])
    else:
        short_side_points = (four_points[3], four_points[2])

    return chess_w, chess_h, long_side_points,short_side_points

def find_distance(p1,p2):
    #compute euclidean distance
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_scale(four_points, M):
    #method used to compute the scale values for height and width, very useful for all the bird view functions

    four_points2 = np.array([four_points], dtype="float32")
    warped_four_points = cv2.perspectiveTransform(four_points2, M)[0]
    '''chess_h = np.sqrt(
        (four_points[0][0] - four_points[3][0]) ** 2 + (four_points[0][1] - four_points[3][1]) ** 2)
    chess_w = np.sqrt(
        (four_points[0][0] - four_points[1][0]) ** 2 + (four_points[0][1] - four_points[1][1]) ** 2)'''
    chess_h = find_distance(four_points[0], four_points[3])
    chess_w = find_distance(four_points[0], four_points[1])

    '''warped_chess_h = np.sqrt(
        (warped_four_points[0][0] - warped_four_points[3][0]) ** 2 + (warped_four_points[0][1] - warped_four_points[3][1]) ** 2)
    warped_chess_w = np.sqrt(
        (warped_four_points[0][0] - warped_four_points[1][0]) ** 2 + (warped_four_points[0][1] - warped_four_points[1][1]) ** 2)'''

    warped_chess_h = find_distance(warped_four_points[0], warped_four_points[3])
    warped_chess_w = find_distance(warped_four_points[0], warped_four_points[1])

    scale_h = chess_h/warped_chess_h
    scale_w = chess_w/warped_chess_w

    return scale_w, scale_h

def find_roi_straight(image,four_points,crop):
    '''method that find coordinates of ROI given an image and the four corner chessboard points, with chessboard in straight position.
    first of all the image will be cropped and after the ROI points will be computed.'''
    image = image[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]
    frame_h = image.shape[0]

    # compute angles
    points1 = [four_points[0], four_points[3], four_points[2]]
    angle1 = find_angle(points1)
    points2 = [four_points[3], four_points[2], four_points[1]]
    angle2 = find_angle(points2)

    left = frame_h/(math.tan(math.radians(int(angle1))))

    right = frame_h/(math.tan(math.radians(int(angle2))))

    new_image = cv2.copyMakeBorder(image, 0,0,int(left),int(right), cv2.BORDER_CONSTANT,BLACK)
    new_h = new_image.shape[0]
    new_w = new_image.shape[1]

    bird_points = [[crop[0][0] + left, crop[0][1]], [crop[0][0] + new_w - right, crop[0][1]],
                  [crop[0][0] + new_w, crop[0][1] + new_h], [crop[0][0], crop[0][1] + new_h]]

    return bird_points, left, right, 0, 0

def find_roi(image,four_points,crop):
    if math.fabs(angleBetween(four_points[3], four_points[2])) < 5 or math.fabs(
            angleBetween(four_points[3], four_points[2])) > 175:
        bird_points, new_left, new_right, new_up, new_down = find_roi_straight(image, four_points, crop)
    else:
        bird_points, new_left, new_right, new_up, new_down = find_roi_askew(image, four_points, crop)
    return bird_points, new_left, new_right, new_up, new_down

def find_roi_askew(image,four_points, crop):
    #this method choose the right way to get the ROI computing angle of chessboard base
    if angleBetween(four_points[3], four_points[2]) < 0 and angleBetween(four_points[0], four_points[1]) < 0:
        #counterclockwise
        bird_points, left, right, up, down = find_roi_counterclockwise(image,four_points,crop)
    else:
        #clockwise
        bird_points, left, right, up, down = find_roi_clockwise(image,four_points,crop)
    return bird_points, left, right, up, down

def find_roi_counterclockwise(image,four_points,crop):
    '''method that find coordinates of ROI given an image and the four corner chessboard points, with chessboard in couterclockwise position.
    first of all the image will be cropped and after the ROI points will be computed.'''

    image = image[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]

    frame_h = image.shape[0]
    frame_w = image.shape[1]

    # compute angles
    points1 = [four_points[3], four_points[0], four_points[1]]
    a = find_angle(points1)

    points2 = [four_points[0], four_points[1], four_points[2]]
    b = find_angle(points2)

    points3 = [four_points[1], four_points[2], four_points[3]]
    c = find_angle(points3)

    points4 = [four_points[2], four_points[3], four_points[0]]
    d = find_angle(points4)


    #find left
    L1 = line([0, 0], [frame_w,0])
    L2 = line([four_points[0][0],four_points[0][1]], [four_points[3][0],four_points[3][1]])

    point_inters1 = intersection(L1,L2)


    points5 = [four_points[0], point_inters1, (frame_w, 0)]
    a_first = find_angle(points5)


    A = (frame_h/(math.sin(math.radians(a))))*math.cos(math.radians(a_first))
    area1 = 0.5 * A * frame_h* math.cos(math.radians(a - a_first))
    left = (2 * area1)/frame_h

    #find up
    L1 = line([0, 0], [0,frame_h])
    L2 = line([four_points[1][0], four_points[1][1]], [four_points[2][0], four_points[2][1]])

    inters_point2 = intersection(L1, L2)

    points6 = [four_points[1], inters_point2, (0, frame_h)]
    b_first = find_angle(points6)

    B = (frame_w / (math.sin(math.radians(b)))) * math.cos(math.radians(b_first))
    area2 = 0.5 * B * frame_w * math.cos(math.radians(b-b_first))
    up = (2 * area2) / frame_w

    #find right
    L1 = line([0, 0], [0, frame_h])
    L2 = line([four_points[2][0], four_points[2][1]], [four_points[3][0], four_points[3][1]])

    inters_point3 = intersection(L1, L2)

    points7 = [four_points[2], inters_point3, (0, 0)]
    c_first = 90-find_angle(points7)

    C = (frame_h / (math.sin(math.radians(c)))) * math.cos(math.radians(c_first))
    area3 = 0.5 * C * frame_h * math.cos(math.radians(c-c_first))
    right = (2 * area3) / frame_h

    #find down
    L1 = line([0, 0], [0,frame_h])
    L2 = line([four_points[0][0], four_points[0][1]], [four_points[1][0], four_points[1][1]])

    inters_point4 = intersection(L1, L2)

    points8 = [four_points[0], inters_point4, (0, 0)]
    d_first = find_angle(points8)

    D = (frame_w / (math.sin(math.radians(d)))) * math.cos(math.radians(d_first))
    area4 = 0.5 * D * frame_w * math.cos(math.radians(d-d_first))
    down = (2 * area4) / frame_w


    X1 = D * math.sin(math.radians(d-d_first))
    Y1 = A * math.sin(math.radians(a-a_first))

    X2 = B * math.sin(math.radians(b-b_first))
    Y2 = C * math.sin(math.radians(c-c_first))

    new_image = cv2.copyMakeBorder(image, int(up), int(down), int(left), int(right), cv2.BORDER_CONSTANT, BLACK)
    new_h = new_image.shape[0]
    new_w = new_image.shape[1]

    bird_points = [[crop[0][0], crop[0][1] + up + Y1], [crop[0][0] + left + X2, crop[0][1]],
                   [crop[0][0] + new_w, crop[0][1] + up + Y2],
                   [crop[0][0] + new_w - right - (frame_w - X1), crop[0][1] + new_h]]

    return bird_points, left, right, up, down

def find_roi_clockwise(image,four_points,crop):
    '''method that find coordinates of ROI given an image and the four corner chessboard points, with chessboard in clockwise position.
    first of all the image will be cropped and after the ROI points will be computed.
    '''
    image = image[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]
    frame_h = image.shape[0]
    frame_w = image.shape[1]

    #compute angles
    points1 = [four_points[2], four_points[3], four_points[0]]
    a = find_angle(points1)

    points2 = [four_points[3], four_points[0], four_points[1]]
    b = find_angle(points2)

    points3 = [four_points[0], four_points[1], four_points[2]]
    c = find_angle(points3)

    points4 = [four_points[1], four_points[2], four_points[3]]
    d = find_angle(points4)

    #find left
    L1 = line([0, 0], [frame_w, 0])
    L2 = line([four_points[3][0], four_points[3][1]], [four_points[2][0], four_points[2][1]])

    inters_point1 = intersection(L1, L2)

    points5 = [four_points[3], inters_point1, (frame_w, 0)]
    a_first = find_angle(points5)

    A = (frame_h / (math.sin(math.radians(a)))) * math.cos(math.radians(a_first))

    area1 = 0.5 * A * frame_h * math.cos(math.radians(a - a_first))
    left = (2 * area1) / frame_h

    # find up
    L1 = line([frame_w, 0], [frame_w, frame_h])
    L2 = line([four_points[0][0], four_points[0][1]], [four_points[3][0], four_points[3][1]])

    inters_point2 = intersection(L1, L2)

    points6 = [four_points[3], inters_point2, (frame_w, frame_h)]
    b_first = find_angle(points6)
    if b_first > 90:
        b_first = 180 - b_first

    B = (frame_w / (math.sin(math.radians(b)))) * math.cos(math.radians(b_first))
    area2 = 0.5 * B * frame_w * math.cos(math.radians(b - b_first))
    up = (2 * area2) / frame_w

    # trovo right
    L1 = line([0, 0], [0, frame_h])
    L2 = line([four_points[1][0], four_points[1][1]], [four_points[2][0], four_points[2][1]])

    inters_point3 = intersection(L1, L2)

    points7 = [four_points[2], inters_point3, (0, 0)]
    c_first = 90 - find_angle(points7)

    C = (frame_h / (math.sin(math.radians(c)))) * math.cos(math.radians(c_first))
    area3 = 0.5 * C * frame_h * math.cos(math.radians(c - c_first))
    right = (2 * area3) / frame_h

    # find down
    L1 = line([0, 0], [0, frame_h])
    L2 = line([four_points[1][0], four_points[1][1]], [four_points[2][0], four_points[2][1]])

    inters_point4 = intersection(L1, L2)


    points8 = [four_points[1], inters_point4, (0, 0)]
    d_first = find_angle(points8)


    D = (frame_w / (math.sin(math.radians(d)))) * math.cos(math.radians(d_first))
    area4 = 0.5 * D * frame_w * math.cos(math.radians(d - d_first))
    down = (2 * area4) / frame_w


    X1 = D * math.sin(math.radians(d - d_first))
    Y1 = A * math.sin(math.radians(a - a_first))

    X2 = B * math.sin(math.radians(b - b_first))
    Y2 = C * math.sin(math.radians(c - c_first))

    new_image = cv2.copyMakeBorder(image, int(up), int(down), int(left), int(right), cv2.BORDER_CONSTANT, BLACK
                                   )
    new_h = new_image.shape[0]
    new_w = new_image.shape[1]
    bird_points = [[crop[0][0] + left + (frame_w - X2), crop[0][1]], [crop[0][0] + new_w, crop[0][1] + up + Y2],
                   [crop[0][0] + left + X1, crop[0][1] + new_h], [crop[0][0], crop[0][1] + up + Y1]]

    return bird_points, left, right, up, down

def getFeetPoints(humanPoints):
    '''method that get only the human feet points from an array with the coordinates of all 25 body parts.
    if one of the two feet is missing, the coordinates will be the same of the other feet.'''

    if len(humanPoints.shape) == 0:
        footPoints = np.ndarray(shape=(1, 2, 2), dtype=float)
        footPoints[0][0][0] = -1
        footPoints[0][0][1] = -1

        footPoints[0][1][0] = -1
        footPoints[0][1][1] = -1
    else:
        footPoints = np.ndarray(shape=(humanPoints.shape[0], 2, 2), dtype=float)
        for i in range(humanPoints.shape[0]):
            for j in range(humanPoints.shape[1]):

                if j == 19:
                    if humanPoints[i][j][0] != 0.0 and humanPoints[i][j][1] != 0.0:
                        footPoints[i][0][0] = humanPoints[i][j][0]
                        footPoints[i][0][1] = humanPoints[i][j][1]
                    elif humanPoints[i][j][0] == 0.0 and humanPoints[i][j][1] == 0.0:
                        footPoints[i][0][0] = humanPoints[i][22][0]
                        footPoints[i][0][1] = humanPoints[i][22][1]
                if j == 22:
                    if humanPoints[i][j][0] != 0.0 and humanPoints[i][j][1] != 0.0:
                        footPoints[i][1][0] = humanPoints[i][j][0]
                        footPoints[i][1][1] = humanPoints[i][j][1]
                    elif humanPoints[i][j][0] == 0.0 and humanPoints[i][j][1] == 0.0:
                        footPoints[i][1][0] = humanPoints[i][19][0]
                        footPoints[i][1][1] = humanPoints[i][19][1]

    return footPoints

def find_angle(points):
    #find angle that forms three points

    a = np.array(points[0])
    b = np.array(points[1])
    c = np.array(points[2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def angleBetween(point1, point2):
    '''
    Calculates the inclination of the line passing through two point.

    '''
    dx = point2[1] - point1[1]
    dy = point2[0] - point1[0]
    arctan = math.atan2(dx, dy)

    return math.degrees(arctan)

def line(p1, p2):
    #create a line given 2 points
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    #find intersection point between two lines
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False
