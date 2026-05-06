import cv2
import numpy as np

# img = cv2.imread('originals/test.jpg', 0)
# img = cv2.imread('originals/image.png', 0)
# img = cv2.imread('originals/f2.png', 0)
img = cv2.imread('originals/img3.png', 0)

if img is None:
    raise ValueError("Изображение не найдено")

# 2. Бинаризация
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Морфология 1: Эрозия (диск r = 3)
kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
erosion = cv2.erode(binary, kernel_disk)

# 4. Морфология 2: Размыкание (крест 5×5)
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_cross)

# 5. Удаление вертикальных объектов 
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 10))
no_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)

cv2.imwrite('processed/binary.png', binary)
cv2.imwrite('processed/erosion.png', erosion)
cv2.imwrite('processed/opening.png', opening)
cv2.imwrite('processed/no_vertical.png', no_vertical)