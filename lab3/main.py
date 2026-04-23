import cv2
import numpy as np
import os

img = cv2.imread("originals/photo.bmp", cv2.IMREAD_GRAYSCALE)

M, N = img.shape

# Дополнение до P x Q
P, Q = 2*M, 2*N
padded = np.zeros((P, Q))
padded[:M, :N] = img

# FFT
F = np.fft.fft2(padded)

# Центрирование
F_shift = np.fft.fftshift(F)

def distance(u, v, center_u, center_v):
    return np.sqrt((u - center_u)**2 + (v - center_v)**2)

def ideal_lpf(D0):
    H = np.zeros((P, Q))
    center_u, center_v = P//2, Q//2

    for u in range(P):
        for v in range(Q):
            if distance(u, v, center_u, center_v) <= D0:
                H[u, v] = 1
    return H

def butterworth_lpf(D0, n):
    H = np.zeros((P, Q))
    center_u, center_v = P//2, Q//2

    for u in range(P):
        for v in range(Q):
            D = distance(u, v, center_u, center_v)
            H[u, v] = 1 / (1 + (D / D0)**(2*n))
    return H

def gaussian_lpf(D0):
    H = np.zeros((P, Q))
    center_u, center_v = P//2, Q//2

    for u in range(P):
        for v in range(Q):
            D = distance(u, v, center_u, center_v)
            H[u, v] = np.exp(-(D**2) / (2 * D0**2))
    return H

def ideal_hpf(D0):
    return 1 - ideal_lpf(D0)

def butterworth_hpf(D0, n):
    return 1 - butterworth_lpf(D0, n)

def gaussian_hpf(D0):
    return 1 - gaussian_lpf(D0)

def apply_filter(H, name):
    # Перемножение
    G = F_shift * H

    # Обратный сдвиг
    G_shift = np.fft.ifftshift(G)

    # Обратное FFT
    g = np.fft.ifft2(G_shift)

    # Действительная часть
    g = np.real(g)

    # Обрезка
    result = g[:M, :N]

    # Сохранение
    os.makedirs("processed", exist_ok=True)
    cv2.imwrite(f"processed/{name}.png", result)

# Спектр
spectrum = np.log(np.abs(F_shift) + 1)
cv2.imwrite("results/spectrum.png", spectrum)

# === Запуск ===

for D0 in [5, 10, 50, 250]:
    apply_filter(ideal_lpf(D0), f"ideal_lpf_{D0}")
    apply_filter(butterworth_lpf(D0, 2), f"butterworth_lpf{D0}")
    apply_filter(gaussian_lpf(D0), f"gaussian_lpf{D0}")

    apply_filter(ideal_hpf(D0), f"ideal_hpf{D0}")
    apply_filter(butterworth_hpf(D0, 2), f"butterworth_hpf{D0}")
    apply_filter(gaussian_hpf(D0), f"gaussian_hpf{D0}")