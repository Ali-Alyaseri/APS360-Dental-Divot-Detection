import os
import shutil
import pandas as pd
from tqdm import tqdm


def convert_to_yolo(size, box):
    """
    Converts Pascal VOC format (xmin, ymin, xmax, ymax) to YOLO format.

    Args:
        size (tuple): A tuple of (image_width, image_height).
        box (tuple): A tuple of (xmin, ymin, xmax, ymax).

    Returns:
        tuple: A tuple of (x_center_norm, y_center_norm, width_norm, height_norm).
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def process_csv_dataset(base_path, combined_path, dataset_name="dental_radiography"):
    """
    Processes a dataset with CSV annotations, filters for a specific class,
    converts to YOLO format, and integrates into a combined dataset.

    Args:
        base_path (str): The path to the root of the source dataset.
        combined_path (str): The path to the 'combined' directory.
        dataset_name (str): The name to prepend to filenames.
    """

    output_images_dir = os.path.join(combined_path, "images")
    output_labels_dir = os.path.join(combined_path, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    TARGET_CLASS = "Cavity"
    YOLO_CLASS_ID = 0

    for subset in ["train", "test", "valid"]:
        print(f"\n--- Processing subset: {subset} ---")
        subset_path = os.path.join(base_path, subset)
        annotations_file = os.path.join(subset_path, "_annotations.csv")

        if not os.path.exists(annotations_file):
            print(f"Annotations file not found for '{subset}'. Skipping.")
            continue

        df = pd.read_csv(annotations_file)

        df_cavity = df[df["class"] == TARGET_CLASS].copy()

        if df_cavity.empty:
            print(f"No '{TARGET_CLASS}' annotations found in '{subset}'.")
            continue

        grouped = df_cavity.groupby("filename")

        for filename, group in tqdm(grouped, desc=f"Processing {subset} images"):
            source_image_path = os.path.join(subset_path, filename)
            if not os.path.exists(source_image_path):
                print(f"Warning: Image file {filename} not found. Skipping.")
                continue

            yolo_labels = []
            for _, row in group.iterrows():
                img_width = int(row["width"])
                img_height = int(row["height"])

                # Bounding box in Pascal VOC format
                box = (
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                )

                # Convert to YOLO format
                bb = convert_to_yolo((img_width, img_height), box)

                yolo_labels.append(
                    f"{YOLO_CLASS_ID} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}"
                )

            new_image_filename = f"{dataset_name}_{filename}"
            new_label_filename = f"{dataset_name}_{os.path.splitext(filename)[0]}.txt"
            dest_image_path = os.path.join(output_images_dir, new_image_filename)
            dest_label_path = os.path.join(output_labels_dir, new_label_filename)

            shutil.copy(source_image_path, dest_image_path)

            with open(dest_label_path, "w") as f:
                f.write("\n".join(yolo_labels))

    print(f"\nProcess complete!")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    radiography_path = os.path.join(current_dir, "..", "dental_radiography")
    combined_dataset_path = os.path.join(current_dir, "..", "smile", "combined")

    process_csv_dataset(
        radiography_path, combined_dataset_path, dataset_name="dental_radiography"
    )
