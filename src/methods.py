import numpy as np

def to_grayscale(image):
    if len(image.shape) == 2: return image
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return np.clip(gray, 0, 255).astype(np.uint8)

def apply_sobel(image):
    padded = np.pad(image, 1, mode='edge').astype(np.float32)
    gx = (padded[0:-2, 2:] - padded[0:-2, 0:-2]) + \
         (padded[1:-1, 2:] * 2 - padded[1:-1, 0:-2] * 2) + \
         (padded[2:, 2:] - padded[2:, 0:-2])
    gy = (padded[2:, 0:-2] - padded[0:-2, 0:-2]) + \
         (padded[2:, 1:-1] * 2 - padded[0:-2, 1:-1] * 2) + \
         (padded[2:, 2:] - padded[0:-2, 2:])
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def threshold_segmentation(image, threshold=128):
    result = np.zeros_like(image)
    result[image >= threshold] = 255
    return result