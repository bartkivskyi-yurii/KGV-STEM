import numpy as np
from src.methods import to_grayscale, threshold_segmentation

def test_to_grayscale():
    test_img = np.ones((3, 3, 3), dtype=np.uint8) * 100
    gray = to_grayscale(test_img)
    assert len(gray.shape) == 2
    assert gray[0, 0] == 100

def test_threshold_segmentation():
    test_img = np.array([[50, 150], [200, 10]], dtype=np.uint8)
    result = threshold_segmentation(test_img, threshold=100)
    assert result[0, 0] == 0
    assert result[0, 1] == 255
    assert result[1, 0] == 255
    assert result[1, 1] == 0