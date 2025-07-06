import os
import shutil
from tqdm import tqdm


def filter_and_integrate(base_path, combined_path, dataset_name="opg_xray"):
    """
    Filters a YOLO dataset for a specific class ID, renames the files by
    prepending the dataset name, and integrates them into a combined dataset.

    Args:
        base_path (str): The path to the root of the source dataset (e.g., 'opg_xray').
        combined_path (str): The path to the 'combined' directory where processed
                             files will be saved.
        dataset_name (str): The name of the dataset to prepend to filenames.
    """

    source_images_dir = os.path.join(base_path, "images")
    source_labels_dir = os.path.join(base_path, "labels")

    output_images_dir = os.path.join(combined_path, "images")
    output_labels_dir = os.path.join(combined_path, "labels")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    target_class_id = "0"

    if not os.path.isdir(source_labels_dir):
        print(f"Error: Source labels directory not found at {source_labels_dir}")
        return

    label_filenames = [f for f in os.listdir(source_labels_dir) if f.endswith(".txt")]

    print(
        f"Filtering and integrating '{dataset_name}' dataset into '{combined_path}'..."
    )

    for label_filename in tqdm(label_filenames, desc=f"Processing {dataset_name}"):
        label_path = os.path.join(source_labels_dir, label_filename)

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Filter lines to keep only those for the target class ID
        filtered_labels = [
            line for line in lines if line.strip().startswith(target_class_id + " ")
        ]

        # Only proceed if the file contains labels for the target class ID
        if not filtered_labels:
            continue

        # Find the corresponding image file (.jpg)
        base_filename = os.path.splitext(label_filename)[0]
        image_filename = base_filename + ".jpg"
        source_image_path = os.path.join(source_images_dir, image_filename)

        if not os.path.exists(source_image_path):
            print(f"Warning: Corresponding image {image_filename} not found. Skipping.")
            continue

        # Create new prepended filenames
        new_image_filename = f"{dataset_name}_{image_filename}"
        new_label_filename = f"{dataset_name}_{label_filename}"

        dest_image_path = os.path.join(output_images_dir, new_image_filename)
        dest_label_path = os.path.join(output_labels_dir, new_label_filename)

        shutil.copy(source_image_path, dest_image_path)

        with open(dest_label_path, "w") as f:
            f.writelines(filtered_labels)

    print(f"Filtering and integration complete!")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    opg_xray_path = os.path.join(current_dir, "..", "opg_xray")
    combined_dataset_path = os.path.join(current_dir, "..", "smile", "combined")

    filter_and_integrate(opg_xray_path, combined_dataset_path, dataset_name="opg_xray")
