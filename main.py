import numpy as np
import cv2
import filterpy.kalman
import filterpy.common
import math
import matplotlib.pyplot as plt


def ukf(x):
    filteredStateX = []
    stateCovarianceHistoryX = []
    for i in range(0, len(x)):
        z1 = [x[i]]
        filter.predict()
        filter.update(z1)

        filteredStateX.append(filter.x)
        stateCovarianceHistoryX.append(filter.P)

    filteredStateX = np.array(filteredStateX)
    stateCovarianceHistoryX = np.array(stateCovarianceHistoryX)
    return filteredStateX, stateCovarianceHistoryX

def measurementFunction(x):
   return np.array([x[0]])

def stateTransitionFunction(x, dt):
   newState = np.zeros(3)
   newState[0] = x[0] + dt * x[1] + ( (dt**2)/2 ) * x[2]
   newState[1] = x[1] + dt * x[2]
   newState[2] = x[2]

   return newState

def filter_kalman (x, dNoise, r, en):
    xx = np.zeros(len(x))  # вектор для хранения оценок перемещений
    P = np.zeros(len(x))  # вектор для хранения дисперсий ошибок оценивания
    xx[0] = x[0]  # первая оценка
    P[0] = dNoise  # дисперсия первой оценки

    # рекуррентное вычисление оценок по фильтру Калмана
    for i in range(1, len(x)):
        Pe = r * r * P[i - 1] + en * en
        P[i] = (Pe * dNoise) / (Pe + dNoise)
        xx[i] = r * xx[i - 1] + P[i] / dNoise * (x[i] - r * xx[i - 1])
    return xx, P

def cam_cal():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((14 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:14, 0:8].T.reshape(-1, 2) * 2000

    objpoints = []
    imgpoints = []
    img = cv2.imread('newChess1.png')
    img = image_resize(img, height=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (14, 8), None)
    print(ret)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (14, 8), corners2, ret)
        viewImage(img)
        rvecs = (0, 0, 0)
        tvecs = (20, 20, 45)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, rvecs,
                                                           tvecs)
        _, _, f, _, k = cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 22.2, 14.7)

        mtx[0][0] = f
        mtx[1][1] = k * f
    return dist, rvecs, tvecs, mtx


def geometricCalibration(RealCoords, ImgCoords):
    if len(RealCoords) == len(ImgCoords):
        success, rotationVector, translationVector = cv2.solvePnP(np.array(RealCoords, dtype="double"),
                                                                  np.array(ImgCoords, dtype="double"),
                                                                  mtx,
                                                                  dist,
                                                                  # tRot, tVec,
                                                                  # None, None,
                                                                  np.float32(tvecs), np.float32(rvecs),
                                                                  cv2.SOLVEPNP_ITERATIVE)
        rtv = rotateTranslationVector(rotationVector, translationVector)
        rm = rotationMatrixToEuler(rotationVector)
    return (rtv, rm)


def rotationMatrixToEuler(rotation_vector):
    np_rodrigues = np.asarray(rotation_vector[:, :], np.float64)
    rot_matrix = cv2.Rodrigues(np_rodrigues)[0]

    y_rot = math.asin(rot_matrix[2][0])
    x_rot = math.acos(rot_matrix[2][2] / math.cos(y_rot))
    z_rot = math.acos(rot_matrix[0][0] / math.cos(y_rot))
    # y_rot_angle = y_rot * (180 / math.pi)
    # x_rot_angle = x_rot * (180 / math.pi)
    # z_rot_angle = z_rot * (180 / math.pi)

    y_rot_angle = z_rot
    x_rot_angle = -x_rot + math.pi
    z_rot_angle = -y_rot

    return np.array([[x_rot_angle], [y_rot_angle], [z_rot_angle]])


def rotateTranslationVector(rotation_vector, translation_vector):
    # Получение реальных координат камеры с учетом её поворота (для отрисовки)
    np_rodrigues = np.asarray(rotation_vector[:, :], np.float64)
    rot_matrix = cv2.Rodrigues(np_rodrigues)[0]

    print((-np.matrix(rot_matrix).T * np.matrix(translation_vector)))
    # сделать без matrix
    tmp = np.array([(-np.matrix(rot_matrix).T * np.matrix(translation_vector))[0][0],
                    (-np.matrix(rot_matrix).T * np.matrix(translation_vector))[1][0],
                    (-np.matrix(rot_matrix).T * np.matrix(translation_vector))[2][0]])

    return np.array([tmp[0][0], tmp[1][0], tmp[2][0]])


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def four_point_transform(image, tl, tr, br, bl):
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32((tl, tr, br, bl)), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    M1 = cv2.getPerspectiveTransform(dst, np.float32((tl, tr, br, bl)))
    return warped, M1


def centerContour(contour, im, color):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(im, (cX, cY), 7, color, -1)
    return cX, cY


fps = 25
time = 0
x = []
y = []
z = []
xa = []
ya = []
za = []
t = []
xlast = 2400

dist, rvecs, tvecs, mtx = cam_cal()
cap = cv2.VideoCapture('1.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # viewImage(frame)
    time = time+1/fps
    frame = image_resize(frame, height=600)
    height, width = frame.shape[:2]
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    thresh = cv2.inRange(hls, (0, np.mean(hls[1]) + 100, 0), (255, 255, 255))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    runway = cv2.inRange(hsv, (0, 0, 0), (255, 75, 255))
    thresh_lowb = cv2.inRange(hls, (0, np.mean(hls[1]) + 70, 0), (255, 255, 255))
    edges = cv2.Canny(thresh, 100, 200)
    lines = cv2.HoughLinesP(
        edges,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    frame1 = frame.copy()
    if lines is not None:
        # viewImage(edges)
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 0.5 or math.fabs(slope) > 2:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
        max_y = frame.shape[0]
        if left_line_x and left_line_y and right_line_x and right_line_y:
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                1
            ))
            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                1
            ))
            min_y = int(np.roots(poly_right - poly_left)) + 65

            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
            cv2.line(frame, (left_x_start, max_y), (left_x_end, min_y), (0, 0, 255), 2)
            cv2.line(frame, (right_x_start, max_y), (right_x_end, min_y), (0, 0, 255), 2)
            # viewImage(frame)
            roiv = np.array([[(left_x_start, max_y), (left_x_end, min_y), (right_x_end, min_y), (right_x_start, max_y)]],
                            dtype=np.int32)
            mask = region_of_interest(thresh_lowb, roiv)
            mask = cv2.bitwise_and(runway, mask)
        else:
            continue
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 10:
            cv2.drawContours(mask, c, -1, 0, -1)
    # viewImage(mask)
    flag = 0
    flag1 = 0
    n = 0
    for i in range(min_y, height):
        if n >= 11 and flag == 0:
            topborder = i
            flag = 1
        if n >= 11 and flag == 1:
            botborder = i
        n = 0
        for j in range(int(poly_left(i)), int(poly_right(i))):
            if mask[i][j] == 255 and mask[i][j + 1] == 0:
                n = n + 1
    # cv2.line(frame1, (int(poly_left(topborder)), topborder), (int(poly_right(topborder)), topborder), (0, 0, 255), 2)
    # cv2.line(frame1, (int(poly_left(botborder)), botborder), (int(poly_right(botborder)), botborder), (0, 0, 255), 2)
    # cv2.line(frame1, (int(poly_left(botborder)), botborder), (left_x_end, min_y), (0, 0, 255), 2)
    # cv2.line(frame1, (int(poly_right(botborder)), botborder), (right_x_end, min_y), (0, 0, 255), 2)
    # viewImage(frame1)
    warped, M = four_point_transform(mask, (left_x_end, min_y), (right_x_end, min_y),
                                     (int(poly_right(botborder)), botborder), (int(poly_left(botborder)), botborder))
    warped1, M = four_point_transform(frame1, (left_x_end, min_y), (right_x_end, min_y),
                                      (int(poly_right(botborder)), botborder), (int(poly_left(botborder)), botborder))
    wh, ww = warped.shape[:2]
    ret, warped = cv2.threshold(warped, 60, 255, 0)
    im2, contours, hierarchy = cv2.findContours(warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(warped1, contours, -1, (255, 0, 0), 2)
    # viewImage(warped1)
    if contours:
        for c in contours:
            try:
                cX, cY = centerContour(c, warped1, (0, 255, 255))
            except ZeroDivisionError:
                print("Контур нулевой площади")
            xbr, ybr, w, h = cv2.boundingRect(c)


            if cv2.contourArea(c) < 300 and cY < 0.8 * wh and (cX > ww / 2 + 10 or cX < ww / 2 - 10) and (h / w < 2):
                cv2.circle(warped1, (cX, cY), 7, (0, 0, 225), -1)
                print(cX, cY)
                coords = M.dot([[cX], [cY], [1]])
                print(coords)

                cv2.circle(frame1, (int(coords[0] / coords[2]), int(coords[1] / coords[2])), 3, (0, 0, 255), -1)
                if cX > ww / 2:
                    if cY < wh / 2:
                        rt = [int(coords[0] / coords[2]), int(coords[1] / coords[2])]
                        cv2.putText(frame1, "rt", tuple(rt),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        rb = [int(coords[0] / coords[2]), int(coords[1] / coords[2])]
                        cv2.putText(frame1, "rb", tuple(rb),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
                else:
                    if cY < wh / 2:
                        lt = [int(coords[0] / coords[2]), int(coords[1] / coords[2])]
                        cv2.putText(frame1, "lt", tuple(lt),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        lb = [int(coords[0] / coords[2]), int(coords[1] / coords[2])]
                        cv2.putText(frame1, "lb", tuple(lb),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
            if cv2.contourArea(c) > 400 and cX > 0.1 * ww and cX < 0.9 * ww and (cX > ww / 2 + 10 or cX < ww / 2 - 10):
                cv2.circle(warped1, (cX, cY), 7, (255, 255, 0), -1)
                print(cX, cY)
                coords = M.dot([[cX], [cY], [1]])
                print(coords)
                cv2.circle(frame1, (int(coords[0] / coords[2]), int(coords[1] / coords[2])), 5, (255, 255, 0), -1)
                if cX > ww / 2:
                    lzr = [int(coords[0] / coords[2]), int(coords[1] / coords[2])]
                    cv2.putText(frame1, "lzr", tuple(lzr),
                                cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
                else:
                    lzl = [int(coords[0] / coords[2]), int(coords[1] / coords[2])]
                    cv2.putText(frame1, "lzl", tuple(lzl),
                                cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
    # viewImage(frame1)
    # viewImage(warped)
    # viewImage(warped1)
    RealCoords = np.float32([[1582.5, 12.75, 0], [1582.5, -12.75, 0], [1418.75, 15.5, 0], [1418.75, -15.5, 0]])
    ImgCoords = np.float32([rb, lb, lzr, lzl])
    rtv, rm = geometricCalibration(RealCoords, ImgCoords)
    print(rtv)
    print(rm)
    cv2.putText(frame, "x: " + str(rtv[0]) + ", y: " + str(rtv[1]) + ", z: " + str(rtv[2]) + " time: " + str(time), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    cv2.putText(frame, "xrot: " + str(math.degrees(rm[0])) + ", yrot: " + str(math.degrees(rm[1])) + ", zrot: " + str(math.degrees(rm[2])), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.imshow('result', frame)
    if abs(rtv[0][0]-xlast) < 75:
        x.append(rtv[0][0])
        y.append(rtv[1][0])
        z.append(rtv[2][0])
        xa.append(math.degrees(rm[0][0]))
        ya.append(math.degrees(rm[1][0]))
        za.append(math.degrees(rm[2][0]))
        t.append(time)
        xlast = rtv[0][0]
        ylast = rtv[1][0]
        zlast = rtv[2][0]


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

fig, ax = plt.subplots(3)
ax[0].plot(t, x)
ax[0].set_title('X(t)')
ax[1].plot(t, y)
ax[1].set_title('Y(t)')
ax[2].plot(t, z)
ax[2].set_title('Z(t)')
# plt.plot(t, x)
# plt.show()
# plt.plot(t, y)
# plt.show()
# plt.plot(t, z)
# plt.show()
ksi = np.zeros(len(x))
for i in range(0, len(x)):
    ksi[i] = x[i]-(-76.5*t[i]+2511)
print(ksi)
print(np.var(ksi))

dt = 0.01
measurementSigma = 0.25
processNoiseVariance = 0.25
points = filterpy.kalman.JulierSigmaPoints(3, kappa=0)
filter = filterpy.kalman.UnscentedKalmanFilter(dim_x = 3,
                                              dim_z = 1,
                                              dt = dt,
                                              hx = measurementFunction,
                                              fx = stateTransitionFunction,
                                              points = points)
filter.Q = filterpy.common.Q_discrete_white_noise(dim=3, dt=dt, var=processNoiseVariance)
filter.R = np.array([[measurementSigma*measurementSigma]])
filter.x = np.array([2410, 0.0, 0.0])
filter.P = np.array([[10.0, 0.0,  0.0],
                    [0.0,  10.0, 0.0],
                    [0.0,  0.0,  10.0]])

filteredStateX, stateCovarianceHistoryX = ukf(x)

filter.x = np.array([12, 0.0, 0.0])

filteredStateY, stateCovarianceHistoryY = ukf(y)

filter.x = np.array([65, 0.0, 0.0])

filteredStateZ, stateCovarianceHistoryZ = ukf(z)

xx, P = filter_kalman(x, 10, 1, 1)
yy, P1 = filter_kalman(y, 10, 1, 1)
zz, P2 = filter_kalman(z, 10, 1, 1)

ax[0].plot(t, xx, color='r')

ax[1].plot(t, yy, color='r')
ax[2].plot(t, zz, color='r')

ax[0].plot([1.32, 5], [2410, 2075], color='g')
ax[1].plot([1.32, 5], [12, 2], color='g')
ax[2].plot([1.32, 5], [65, 32], color='g')

plt.show()


fig, ax = plt.subplots(3)
ax[0].plot(t, x)
ax[0].set_title('X(t)')
ax[1].plot(t, y)
ax[1].set_title('Y(t)')
ax[2].plot(t, z)
ax[2].set_title('Z(t)')
ax[0].plot(t, filteredStateX[:, 0], color='b')
ax[1].plot(t, filteredStateY[:, 0], color='b')
ax[2].plot(t, filteredStateZ[:, 0], color='b')
ax[0].plot([1.32, 5], [2410, 2075], color='g')
ax[1].plot([1.32, 5], [12, 2], color='g')
ax[2].plot([1.32, 5], [65, 32], color='g')

plt.show()

ksi1 = abs(x-xx)

ksi2 = abs(x-filteredStateX[:,0])

plt.plot(t, P)
plt.show()

plt.plot(t, abs(ksi))
plt.plot(t, ksi1, color='r')
plt.plot(t, ksi1, color='g')
plt.show()

for i in range(0, len(x)):
    ksi[i] = filteredStateX[i,0] -(-76.5*t[i]+2511)
print(ksi)
print(np.var(ksi))

fig, ax1 = plt.subplots(3)
ax1[0].plot(t, xa)
ax1[0].set_title('X Rotation(t)')
ax1[1].plot(t, ya)
ax1[1].set_title('Y Rotation(t)')
ax1[2].plot(t, za)
ax1[2].set_title('Z Rotation(t)')
xx, P = filter_kalman(xa, 10, 1, 1)
yy, P1 = filter_kalman(ya, 10, 1, 1)
zz, P2 = filter_kalman(za, 10, 1, 1)
ax1[0].plot(t, xx, color='r')
ax1[1].plot(t, yy, color='r')
ax1[2].plot(t, zz, color='r')
ax1[0].plot([1.32, 5], [0, 0], color='g')
ax1[1].plot([1.32, 5], [90, 90], color='g')
ax1[2].plot([1.32, 5], [85, 90], color='g')
plt.show()

filter.x = np.array([0, 0.0, 0.0])

filteredStateX, stateCovarianceHistoryX = ukf(xa)

filter.x = np.array([90, 0.0, 0.0])

filteredStateY, stateCovarianceHistoryY = ukf(ya)

filter.x = np.array([85, 0.0, 0.0])

filteredStateZ, stateCovarianceHistoryZ = ukf(za)

fig, ax1 = plt.subplots(3)
ax1[0].plot(t, xa)
ax1[0].set_title('X Rotation(t)')
ax1[1].plot(t, ya)
ax1[1].set_title('Y Rotation(t)')
ax1[2].plot(t, za)
ax1[2].set_title('Z Rotation(t)')
ax[0].plot(t, filteredStateX[:, 0], color='b')
ax[1].plot(t, filteredStateY[:, 0], color='b')
ax[2].plot(t, filteredStateZ[:, 0], color='b')
ax1[0].plot([1.32, 5], [0, 0], color='g')
ax1[1].plot([1.32, 5], [90, 90], color='g')
ax1[2].plot([1.32, 5], [85, 90], color='g')
plt.show()
