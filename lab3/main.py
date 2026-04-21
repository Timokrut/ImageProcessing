import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Any

ORIGINAL_JPG_PATH = 'originals/photo.jpg'
ORIGINAL_BMP_PATH = 'originals/photo.bmp'

class Photo:
    def __init__(self, path: str) -> None:
        self.path = path 
        self.img = cv2.imread(path)
        _, self.prefix = os.path.splitext(path)

        if self.img is None:
            raise Exception(f"Couldn't find file at {path}")

        dimensions = self.img.shape 
        self.height = dimensions[0]
        self.width = dimensions[1]
        self.channels = dimensions[2] if len(dimensions) == 3 else 1
        self.file_size = os.path.getsize(self.path)

    def __str__(self) -> str:
        return f"Image stats: \nHeight: {self.height}\nWidth: {self.width} \nChannels: {self.channels} \nFile size: {self.file_size}"

    def make_halftone(self) -> tuple[Any, str]: 
        halftone_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        self.hf_img = halftone_img
        os.makedirs(f"processed/halftone/{self.prefix[1:]}", exist_ok=True)
        cv2.imwrite(f"processed/halftone/{self.prefix[1:]}/halftone{self.prefix}", halftone_img)

        return (halftone_img, f"processed/halftone/{self.prefix[1:]}/halftone{self.prefix}")

    def get_fourier_spectrum(self):
        f = np.fft.fft2(self.hf_img)
        fshift = np.fft.fftshift(f)

        magnitude = 20 * np.log(np.abs(fshift) + 1)

        os.makedirs(f"processed/spectrum/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/spectrum/{self.prefix[1:]}/spectrum{self.prefix}"

        cv2.imwrite(path, magnitude)

        self.fshift = fshift
        return (magnitude, path)
    
    def _distance(self, i, j, crow, ccol):
        return np.sqrt((i - crow)**2 + (j - ccol)**2)
    
    def ideal_lpf(self, D0):
        rows, cols = self.hf_img.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols), np.uint8)

        for i in range(rows):
            for j in range(cols):
                if self._distance(i, j, crow, ccol) <= D0:
                    mask[i, j] = 1

        return self._apply_filter(mask, f"ideal_lpf{D0}", "ideal_lpf")
    
    def butterworth_lpf(self, D0, n):
        rows, cols = self.hf_img.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                D = self._distance(i, j, crow, ccol)
                mask[i, j] = 1 / (1 + (D / D0)**(2*n))

        return self._apply_filter(mask, f"butter_lpf_D0_{D0}_n_{n}", "butterworth_lpf")
    
    def gaussian_lpf(self, D0=50):
        rows, cols = self.hf_img.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                D = self._distance(i, j, crow, ccol)
                mask[i, j] = np.exp(-(D**2) / (2 * D0**2))

        return self._apply_filter(mask, f"gaussian_lpf{D0}", "gaussian_lpf")
    
    def ideal_hpf(self, D0):
        mask_lpf, _ = self.ideal_lpf(D0)
        return self._apply_filter(1 - mask_lpf, f"ideal_hpf{D0}", "ideal_hpf")

    def butterworth_hpf(self, D0, n):
        mask_lpf, _ = self.butterworth_lpf(D0, n)
        return self._apply_filter(1 - mask_lpf, f"butter_hpf_D0_{D0}_n_{n}", "butterworth_hpf")

    def gaussian_hpf(self, D0=50):
        mask_lpf, _ = self.gaussian_lpf(D0)
        return self._apply_filter(1 - mask_lpf, f"gaussian_hpf{D0}", "gaussian_hpf")

    def _apply_filter(self, mask, name, folder):
        fshift = self.fshift

        filtered = fshift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
        img_back = np.abs(img_back)

        os.makedirs(f"processed/{folder}/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/{folder}/{self.prefix[1:]}/{name}{self.prefix}"

        cv2.imwrite(path, img_back)

        return (img_back, path)


if __name__ == "__main__":
    # jpg_photo = Photo(ORIGINAL_JPG_PATH)
    bmp_photo = Photo(ORIGINAL_BMP_PATH)

    # jpg_photo.make_halftone()
    bmp_photo.make_halftone()

    # jpg_photo.get_fourier_spectrum()
    bmp_photo.get_fourier_spectrum()

    D0_values = [5, 10, 50, 250]
    
    for D0 in D0_values:
        # НЧ
        # jpg_photo.ideal_lpf(D0)
        bmp_photo.ideal_lpf(D0)
        # jpg_photo.butterworth_lpf(D0, 2)
        bmp_photo.butterworth_lpf(D0, 2)
        # jpg_photo.gaussian_lpf(D0)
        bmp_photo.gaussian_lpf(D0)

        # ВЧ
        # jpg_photo.ideal_hpf(D0)
        bmp_photo.ideal_hpf(D0)
        # jpg_photo.butterworth_hpf(D0, 2)
        bmp_photo.butterworth_hpf(D0, 2)
        # jpg_photo.gaussian_hpf(D0)
        bmp_photo.gaussian_hpf(D0)