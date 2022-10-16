from skimage import io
import cv2
import numpy as np

url = "https://github.com/andrewiva99/ComputerVision/blob/main/whiteballssample.jpg?raw=true"


def count_balls(url):
    img = io.imread(url)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    thresh = 160
    img_binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    circles = cv2.HoughCircles(img_binary, cv2.HOUGH_GRADIENT, 1, 60,
                               param1=50, param2=13, minRadius=0, maxRadius=80)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 3)

    rad = [el[2] for el in circles[0, :]]
    count = len(rad)
    print("Количество шаров:", count)
    print("Средний радиус:", int(np.mean(rad)))
    print("Дисперсия:", round(np.var(rad), 3))

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    count_balls(url)