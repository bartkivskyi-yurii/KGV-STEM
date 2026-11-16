from src.methods import to_grayscale, apply_sobel, threshold_segmentation

def run_pipeline(image, pipeline_name, config):
    if pipeline_name == "student1_p1":
        # Учасник 1: Виявлення замаскованої техніки (контурний аналіз)
        s1 = to_grayscale(image)
        s2 = apply_sobel(s1)
        s3 = threshold_segmentation(s2, config.get("threshold", 100))
        return s3
    else:
        raise ValueError(f"Невідомий пайплайн: {pipeline_name}")