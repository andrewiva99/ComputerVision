import numpy as np
import cv2
from skimage import io


# Заполнение краев массива нулями, для реализации техники Padding
def fill_zeros(src, d):
    l_r, l_c = src.shape
    z_h = np.zeros((l_r, d))
    z_v = np.zeros((d, l_c + 2 * d))
    res = np.hstack((src, z_h))
    res = np.hstack((z_h, res))
    res = np.vstack((z_v, res))
    res = np.vstack((res, z_v))
    return res


# Функция активации ReLU
def activ_relu(matrx):
    l_r, l_c = matrx.shape
    res = np.zeros((l_r, l_c))
    for i in range(l_r):
        for j in range(l_c):
            if matrx[i, j] > 255:
                res[i, j] = 255
            elif matrx[i, j] < 0:
                res[i, j] = 0
            else:
                res[i, j] = matrx[i, j]
    return np.asarray(res, dtype=np.uint8)


# Применение фильтра к каналу
def filter2D(src, krnl, border_zero=False):
    l_r, l_c = src.shape
    d = len(krnl) // 2
    if border_zero:
        src = fill_zeros(src, d)
    else:
        lk_r, lk_c = krnl.shape
        l_r, l_c = (l_r - lk_r) + 1, (l_c - lk_c) + 1
    res = np.zeros((l_r, l_c))

    for i in range(d, l_r + 1):
        for j in range(d, l_c + 1):
            tmp = src[i - d: i + d + 1, j - d: j + d + 1] * krnl
            cnv = tmp.sum()
            res[i - d, j - d] = cnv
    res = activ_relu(np.array(res))
    return res


# Свертка; функция активации ReLU встроена
def conv(src, filters, bias, border_zero=False):
    n_f, lk_r, lk_c, _ = filters.shape  # число фильтров
    n_ch, l_r, l_c = src.shape  # число каналов и размерность
    if border_zero == False:
        l_r, l_c = (l_r - lk_r) + 1, (l_c - lk_c) + 1
    res = []
    for i in range(n_f):
        res_cf = np.zeros((l_r, l_c))
        for j in range(n_ch):
            curr = filter2D(src[j], filters[i, j], border_zero)
            res_cf += curr
        res_f = res_cf + bias[i]
        res_f = activ_relu(res_f)
        res.append(res_f)
    return np.asarray(res, dtype = np.uint8)


# Max pooling на одном канале src
# Окно размерности (2*2), шаг пулинга = 2
def max_pool(src):
    l_r, l_c = src.shape
    res = []
    c = 2
    r = 2
    for i in range(0, l_r, 2):
        curr = []
        if i == l_r - 1:
            r = 1
        for j in range(0, l_c, 2):
            if j == l_c - 1:
                c = 1
            curr.append(np.max(src[i:i + 2, j:j + c]))
            c = 2
        res.append(curr)
    return np.array(res)


# Max pooling на массиве каналов
def pooling(chnls):
    n_ch = len(chnls)
    res = []
    for i in range(n_ch):
        res.append(max_pool(chnls[i]))
    return np.array(res)

# функция softmax
def softmax(chnls):
    curr = np.float64(chnls).reshape(-1)
    res = np.exp(curr)
    res = res/np.sum(res)
    return res



# "Генерация" n фильтров размерности (w * h * d)
def filters(n, w, h, d):
    res = []
    for i in range(n):
        fltr = [None]
        for j in range(d):
            curr = np.random.randint(-2, 2, (h, w))
            if j == 0:
                fltr[0] = curr
            else:
                fltr.append(curr)
        res.append(fltr)
    return np.array(res)

# Вектор смещения
def bias(n):
    return np.random.randint(-2, 2, n)


url = "https://github.com/andrewiva99/ComputerVision/blob/main/popugai.jpg?raw=true"
src = io.imread(url)
src = np.array(cv2.split(src))


#--------дз №2 (3 фильтра)---------
fltr = filters(3, 3, 3, 3)
bs = bias(3)
# ReLu встроена в conv
cnv = conv(src, fltr, bs, border_zero=True)
max_pooling = pooling(cnv)
res = softmax(max_pooling)


#--------дз №3 (5 фильтров)--------
'''
fltr = filters(5, 3, 3, 3)
bs = bias(5)
# ReLu встроена в conv
cnv = conv(src, fltr, bs, border_zero=True)
max_pooling = pooling(cnv)
res = softmax(max_pooling)
'''