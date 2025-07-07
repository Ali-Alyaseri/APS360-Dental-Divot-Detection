import os
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


def draw_bounding_boxes(csv_path, images_dir, output_dir):
    """
    Reads a CSV file with bounding box data, draws the boxes on the corresponding
    images, and saves them to a new directory, showing a progress bar.

    Args:
        csv_path (str): The path to the input CSV file.
        images_dir (str): The path to the directory containing the original images.
        output_dir (str): The path to the directory where the new images will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)

    grouped = df.groupby("image_filename")
    for image_filename, group in tqdm(grouped, desc="Processing Images"):
        image_path = os.path.join(images_dir, image_filename)

        with Image.open(image_path) as img:
            img_color = img.convert("RGB")
            draw = ImageDraw.Draw(img_color)

            for _, row in group.iterrows():
                x1, y1, x2, y2 = (
                    row["x_min"],
                    row["y_min"],
                    row["x_max"],
                    row["y_max"],
                )

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            output_path = os.path.join(output_dir, image_filename)
            img_color.save(output_path)


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    combined_dir = os.path.join(current_script_dir, "..", "smile", "combined")

    CSV_FILE = os.path.join(combined_dir, "meta", "bounding_boxes.csv")
    INPUT_IMAGES_DIR = os.path.join(combined_dir, "images")
    OUTPUT_IMAGES_DIR = os.path.join(combined_dir, "images_with_bboxes")
    draw_bounding_boxes(CSV_FILE, INPUT_IMAGES_DIR, OUTPUT_IMAGES_DIR)
