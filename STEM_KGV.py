import os
import json
import argparse
import numpy as np
from PIL import Image

from src.pipelines import run_pipeline
from src.metrics import calculate_psnr, calculate_mse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "images")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "results")
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.json")

def main():
    parser = argparse.ArgumentParser(description="STEM Проєкт - Військова техніка")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Папка з оригіналами")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Головна папка результатів")
    # Поки що доступний лише пайплайн 1-го учасника
    parser.add_argument("--pipeline", type=str, required=True, 
                        choices=["student1_p1"], 
                        help="Який пайплайн запустити")
    
    args = parser.parse_args()

    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    pipeline_config = config["pipelines"][args.pipeline]
    output_dir = os.path.join(args.output, args.pipeline)
    os.makedirs(output_dir, exist_ok=True)

    metrics_log = []
    
    if not os.path.exists(args.input):
        print(f"Помилка: Папку із зображеннями не знайдено ({args.input})")
        return

    for img_name in os.listdir(args.input):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(args.input, img_name)
        image = np.array(Image.open(img_path))

        result_image = run_pipeline(image, args.pipeline, pipeline_config)

        out_path = os.path.join(output_dir, f"processed_{img_name}")
        mode = "L" if len(result_image.shape) == 2 else "RGB"
        Image.fromarray(result_image, mode=mode).save(out_path)

        try:
            from src.methods import to_grayscale
            orig_gray = to_grayscale(image)
            res_gray = to_grayscale(result_image)
            mse = calculate_mse(orig_gray, res_gray)
            psnr = calculate_psnr(orig_gray, res_gray)
        except Exception:
            mse, psnr = float('inf'), 0

        metrics_log.append({"image": img_name, "MSE": round(mse, 4), "PSNR": round(psnr, 4)})
        print(f"[{args.pipeline}] Оброблено {img_name} | PSNR: {psnr:.2f}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=4)
    print(f"Метрики збережено у {metrics_path}")

if __name__ == "__main__":
    main()