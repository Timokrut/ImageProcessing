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

    def log_transform(self, gray_img) -> tuple[Any, str]:
        img_norm = gray_img / 255.0

        c = 255 / np.log(1 + np.max(img_norm))
        log_img = c * np.log(1 + img_norm)

        log_img = np.uint8(log_img)

        os.makedirs(f"processed/log/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/log/{self.prefix[1:]}/log{self.prefix}"
        cv2.imwrite(path, log_img)

        return (log_img, path)
    
    def gamma_transform(self, gray_img, gamma: float) -> tuple[Any, str]:
        img_norm = gray_img / 255.0

        gamma_img = np.power(img_norm, gamma)
        gamma_img = np.uint8(gamma_img * 255)

        os.makedirs(f"processed/gamma/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/gamma/{self.prefix[1:]}/gamma_{gamma}{self.prefix}"
        cv2.imwrite(path, gamma_img)

        return (gamma_img, path)

    def add_exp_noise(self, gray_img, scale: float) -> tuple[Any, str]:
        noise = np.random.exponential(scale, gray_img.shape)

        noisy_img = gray_img + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_img = np.uint8(noisy_img)

        self.noise_img = noisy_img

        os.makedirs(f"processed/noise/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/noise/{self.prefix[1:]}/exp_noise{self.prefix}"
        cv2.imwrite(path, noisy_img)

        return (noisy_img, path)

    def plot_histogram(self, img, title: str):
        plt.figure()
        plt.hist(img.ravel(), bins=256, range=[0,256])
        plt.title(title)
        plt.xlabel("Intensity")
        plt.ylabel("Pixels")
        plt.xlim(0, 255)
        os.makedirs(f"processed/graph/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/graph/{self.prefix[1:]}/histogram.jpg"
        plt.savefig(path)

    def mean_filter(self, size: int) -> tuple[Any, str]:
        filtered = cv2.blur(self.noise_img, (size, size))

        os.makedirs(f"processed/mean_filter/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/mean_filter/{self.prefix[1:]}/mean_{size}{self.prefix}"
        cv2.imwrite(path, filtered)

        return (filtered, path)
    
    def sharpen(self) -> tuple[Any, str]:
        matrix = np.array([[0,1,0],
                          [1,-4,1],
                          [0,1,0]])

        sharp = cv2.filter2D(self.hf_img, -1, matrix)

        os.makedirs(f"processed/sharpen/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/sharpen/{self.prefix[1:]}/sharpen{self.prefix}"
        cv2.imwrite(path, sharp)

        return (sharp, path)

    def roberts(self):
        matrix_x = np.array([[1, 0],
                            [0, -1]])

        matrix_y = np.array([[0, 1],
                            [-1, 0]])

        gx = cv2.filter2D(self.hf_img, -1, matrix_x)
        gy = cv2.filter2D(self.hf_img, -1, matrix_y)

        roberts = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        roberts = np.uint8(roberts)
        
        os.makedirs(f"processed/roberts/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/roberts/{self.prefix[1:]}/roberts{self.prefix}"
        cv2.imwrite(path, roberts)
        return roberts, path
    
    def prewitt(self):
        matrix_x = np.array([[-1,0,1],
                            [-1,0,1],
                            [-1,0,1]])

        matrix_y = np.array([[1,1,1],
                            [0,0,0],
                            [-1,-1,-1]])

        gx = cv2.filter2D(self.hf_img, -1, matrix_x)
        gy = cv2.filter2D(self.hf_img, -1, matrix_y)

        prewitt = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        prewitt = np.uint8(prewitt)

        os.makedirs(f"processed/prewitt/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/prewitt/{self.prefix[1:]}/prewitt{self.prefix}"
        cv2.imwrite(path, prewitt)
        
        return prewitt, path 
    
    def sobel(self):
        matrix_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

        matrix_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])
        
        gx = cv2.filter2D(self.hf_img, -1, matrix_x)
        gy = cv2.filter2D(self.hf_img, -1, matrix_y)

        
        sobel = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        sobel = np.uint8(sobel)

        os.makedirs(f"processed/sobel/{self.prefix[1:]}", exist_ok=True)
        path = f"processed/sobel/{self.prefix[1:]}/sobel{self.prefix}"
        cv2.imwrite(path, sobel)
        return sobel, path


if __name__ == "__main__":
    jpg_photo = Photo(ORIGINAL_JPG_PATH)
    bmp_photo = Photo(ORIGINAL_BMP_PATH)

    hf_jpg, _ = jpg_photo.make_halftone()
    hf_bmp, _ = bmp_photo.make_halftone()

    jpg_photo.log_transform(hf_jpg)
    bmp_photo.log_transform(hf_bmp)

    for gamma in [0.1, 0.45, 5]:
        bmp_photo.gamma_transform(hf_bmp, gamma)
        jpg_photo.gamma_transform(hf_jpg, gamma)

    # Доп задание
    # gamma01, _ = bmp_photo.gamma_transform(bmp_photo.hf_img.astype(np.float32), 0.1)
    # gamma10, _ = bmp_photo.gamma_transform(gamma01, 10)
    
    noise_jpg, _ = jpg_photo.add_exp_noise(hf_jpg, 20)
    noise_bmp, _ = bmp_photo.add_exp_noise(hf_bmp, 20)

    jpg_photo.plot_histogram(noise_jpg, 'Halftone + Noise image histogram')
    bmp_photo.plot_histogram(hf_bmp, 'Halftone image histogram')

    for k in [1, 3, 9, 15]:
        jpg_photo.mean_filter(k)
        bmp_photo.mean_filter(k)

    jpg_photo.sharpen()
    bmp_photo.sharpen()

    bmp_photo.roberts()
    bmp_photo.prewitt()
    bmp_photo.sobel()



    