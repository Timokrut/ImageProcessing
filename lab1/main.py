import cv2
import numpy as np
import os
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

    def calculate_compression_rate(self) -> float:
        # Get the image data type (e.g., uint8, uint16, float32)
        dtype = self.img.dtype
        # Get the size of the data type in bytes (e.g., uint8 is 1 byte, uint16 is 2 bytes)
        bytes_per_channel = dtype.itemsize
        
        # ??? per channel / per pixel??
        bits_per_channel = bytes_per_channel * 8 * self.channels

        return ((self.width * self.height * bits_per_channel) / 8 ) / self.file_size 
        
    def split_to_channels(self) -> tuple[str, str, str]:
        b_channel, g_channel, r_channel = cv2.split(self.img)

        k = np.zeros_like(b_channel)

        b_channel = cv2.merge([b_channel, k, k])
        g_channel = cv2.merge([k, g_channel, k])
        r_channel = cv2.merge([k, k, r_channel])

        cv2.imwrite(f"processed/RGB_channels/{self.prefix[1:]}/r{self.prefix}", r_channel)
        cv2.imwrite(f"processed/RGB_channels/{self.prefix[1:]}/g{self.prefix}", g_channel)
        cv2.imwrite(f"processed/RGB_channels/{self.prefix[1:]}/b{self.prefix}", b_channel)

        return (f"processed/RGB_channels/{self.prefix[1:]}/b{self.prefix}", f"processed/RGB_channels/{self.prefix[1:]}/g{self.prefix}", f"processed/RGB_channels/{self.prefix[1:]}/r{self.prefix}") 

    def make_halftone(self) -> tuple[Any, str]: 
        halftone_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        cv2.imwrite(f"processed/halftone/{self.prefix[1:]}/halftone{self.prefix}", halftone_img)

        return (halftone_img, f"processed/halftone/{self.prefix[1:]}/halftone{self.prefix}")

    def binary_by_percent(self, halftone_img: Any, percent: int):
        if not(0 <= percent <= 100): 
            raise Exception("Wrong percent given")

        threshold_value = int(255 * percent / 100)

        _, bin = cv2.threshold(halftone_img, threshold_value, 255, cv2.THRESH_BINARY)
        return bin

    def binarize(self, arg1=25, arg2=50, arg3=75) -> tuple[str, str, str]:
        halftone_img = self.make_halftone()[0] 
        binary_1 = self.binary_by_percent(halftone_img, arg1)
        binary_2 = self.binary_by_percent(halftone_img, arg2)
        binary_3 = self.binary_by_percent(halftone_img, arg3)

        cv2.imwrite(f"processed/binarized/{self.prefix[1:]}/bin{arg1}{self.prefix}", binary_1)
        cv2.imwrite(f"processed/binarized/{self.prefix[1:]}/bin{arg2}{self.prefix}", binary_2)
        cv2.imwrite(f"processed/binarized/{self.prefix[1:]}/bin{arg3}{self.prefix}", binary_3)

        return (f"processed/binarized/{self.prefix[1:]}/bin{arg1}{self.prefix}", f"processed/binarized/{self.prefix[1:]}/bin{arg2}{self.prefix}", f"processed/binarized/{self.prefix[1:]}/bin{arg3}{self.prefix}")

    def mirror(self, flip_code=0) -> tuple[Any, str]:
        # flipCode = 0: Flips the image around the x-axis (vertical mirror).
        # flipCode > 0 (e.g., 1): Flips the image around the y-axis (horizontal mirror).
        # flipCode < 0 (e.g., -1): Flips the image around both axes

        flipped_img = cv2.flip(self.img, flip_code)
        cv2.imwrite(f'processed/mirrored/{self.prefix[1:]}/mirrored{self.prefix}', flipped_img)

        return (flipped_img, f"processed/mirrored/{self.prefix[1:]}/mirrored{self.prefix}")

    def rotate(self, value=cv2.ROTATE_90_CLOCKWISE) -> tuple[Any, str]:
        rotated_img = cv2.rotate(self.img, value)
        cv2.imwrite(f'processed/rotated/{self.prefix[1:]}/rotated{self.prefix}', rotated_img)

        return (rotated_img, f"processed/rotated/{self.prefix[1:]}/rotated{self.prefix}")


    def block_discretization(self, block_size: int) -> tuple[Any, str]:
        halftone_img = self.make_halftone()[0] 

        new_h = self.height // block_size
        new_w = self.width // block_size
        
        resized = cv2.resize(halftone_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.resize(resized, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(f'processed/discretized/{self.prefix[1:]}/discretized{block_size}{self.prefix}', result)

        return (result, f"processed/discretized/{self.prefix[1:]}/discretized{block_size}{self.prefix}")

    def quantize(self, level: int) -> tuple[Any, str]:
        halftone_img = self.make_halftone()[0] 

        step = 256 // level
        quantized = (halftone_img // step) * step + step // 2
        
        cv2.imwrite(f'processed/quantized/{self.prefix[1:]}/quantized{level}{self.prefix}', quantized)
        return (quantized, f"processed/quantized/{self.prefix[1:]}/quantized{level}{self.prefix}")

    def cut_rectangle_in_middle(self, x: int, y: int) -> tuple[Any, str]:
        halftone_img = self.make_halftone()[0] 
        center_x = self.width // 2 
        center_y = self.height // 2

        x_start = center_x - x // 2
        x_end = center_x + x // 2 
        y_start = center_y - y // 2 
        y_end = center_y + y // 2

        cropped = halftone_img[y_start:y_end, x_start:x_end]

        cv2.imwrite(f'processed/cropped/{self.prefix[1:]}/cropped{self.prefix}', cropped)
        return (cropped, f"processed/cropped/{self.prefix[1:]}/cropped{self.prefix}")

    def resize(self, image, x: int, y: int) -> tuple:
        resized_linear = cv2.resize(image, (x, y), interpolation=cv2.INTER_LINEAR)
        resized_cubic = cv2.resize(image, (x, y), interpolation=cv2.INTER_CUBIC)
        resized_area = cv2.resize(image, (x, y), interpolation=cv2.INTER_AREA)
        resized_nearest = cv2.resize(image, (x, y), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f'processed/resized/{self.prefix[1:]}/resized_linear{self.prefix}', resized_linear)
        cv2.imwrite(f'processed/resized/{self.prefix[1:]}/resized_cubic{self.prefix}', resized_cubic)
        cv2.imwrite(f'processed/resized/{self.prefix[1:]}/resized_area{self.prefix}', resized_area)
        cv2.imwrite(f'processed/resized/{self.prefix[1:]}/resized_nearest{self.prefix}', resized_nearest)

        return (resized_linear, f"processed/resized/{self.prefix[1:]}/resized_linear{self.prefix}", resized_cubic, f"processed/resized/{self.prefix[1:]}/resized_cubic{self.prefix}", resized_area, f"processed/resized/{self.prefix[1:]}/resized_area{self.prefix}", resized_nearest, f"processed/resized/{self.prefix[1:]}/resized_nearest{self.prefix}")

if __name__ == "__main__":
    jpg_photo = Photo(ORIGINAL_JPG_PATH)
    bmp_photo = Photo(ORIGINAL_BMP_PATH)
    #test_photo = Photo('originals/test.bmp')

    print(jpg_photo.calculate_compression_rate())
    print(bmp_photo.calculate_compression_rate())

    jpg_photo.split_to_channels()
    bmp_photo.split_to_channels()

    jpg_photo.make_halftone()
    bmp_photo.make_halftone()

    jpg_photo.binarize()
    bmp_photo.binarize()
    # test_photo.binarize()

    jpg_photo.mirror()
    bmp_photo.mirror()

    jpg_photo.rotate()
    bmp_photo.rotate()

    for block_size in [5, 10, 20, 50]:
        jpg_photo.block_discretization(block_size) 
        bmp_photo.block_discretization(block_size)

    for level in [4, 16, 32, 64, 128]:
        jpg_photo.quantize(level)
        bmp_photo.quantize(level)

    cut_jpg = jpg_photo.cut_rectangle_in_middle(100, 100)[0]
    cut_bmp = bmp_photo.cut_rectangle_in_middle(100, 100)[0]

    jpg_photo.resize(cut_jpg, 300, 300)
    bmp_photo.resize(cut_bmp, 300, 300)
