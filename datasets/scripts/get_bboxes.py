import os
import csv
from PIL import Image
from tqdm import tqdm


def export_full_yolo_data_to_csv(images_dir, labels_dir, output_csv_path):
    """
    Reads YOLO files, calculates full bounding box coordinates (top-left,
    bottom-right) and dimensions, and exports everything to a CSV file.

    Args:
        images_dir (str): Path to the directory containing images.
        labels_dir (str): Path to the directory containing YOLO .txt labels.
        output_csv_path (str): Path for the output CSV file.
    """

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    try:
        with open(output_csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            header = [
                "image_filename",
                "image_width",
                "image_height",
                "class_id",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "bbox_width",
                "bbox_height",
            ]
            csv_writer.writerow(header)

            label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
            print(f"Found {len(label_files)} label files. Processing...")

            for label_file in tqdm(label_files, desc="Exporting bounding boxes"):
                image_name_base = os.path.splitext(label_file)[0]

                image_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    potential_path = os.path.join(images_dir, f"{image_name_base}{ext}")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break

                if not image_path:
                    continue

                label_path = os.path.join(labels_dir, label_file)

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            (
                                class_id,
                                x_center_norm,
                                y_center_norm,
                                width_norm,
                                height_norm,
                            ) = map(float, parts)

                            # De-normalize dimensions and coordinates
                            bbox_width = width_norm * img_width
                            bbox_height = height_norm * img_height
                            x_center = x_center_norm * img_width
                            y_center = y_center_norm * img_height

                            # Calculate coordinates
                            x_min = x_center - (bbox_width / 2)
                            y_min = y_center - (bbox_height / 2)
                            x_max = x_center + (bbox_width / 2)
                            y_max = y_center + (bbox_height / 2)

                            row = [
                                os.path.basename(image_path),
                                img_width,
                                img_height,
                                int(class_id),
                                int(round(x_min)),
                                int(round(y_min)),
                                int(round(x_max)),
                                int(round(y_max)),
                                int(round(bbox_width)),
                                int(round(bbox_height)),
                            ]
                            csv_writer.writerow(row)

        print(f"\nAll bounding box data has been exported to '{output_csv_path}'")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIRECTORY = os.path.join(current_dir, "..", "smile", "combined", "images")
    LABELS_DIRECTORY = os.path.join(current_dir, "..", "smile", "combined", "labels")
    OUTPUT_CSV_FILE = os.path.join(
        current_dir, "..", "smile", "combined", "meta", "bounding_boxes.csv"
    )

    export_full_yolo_data_to_csv(IMAGES_DIRECTORY, LABELS_DIRECTORY, OUTPUT_CSV_FILE)
