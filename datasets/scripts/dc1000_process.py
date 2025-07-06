import os
import cv2
import shutil
from tqdm import tqdm


def process_and_integrate_dataset(base_path, combined_path, dataset_name="dc1000"):
    """
    Processes image segmentation masks, converts them to YOLO format, and adds
    them to a global 'combined' dataset directory with a progress bar.

    Args:
        base_path (str): The path to the root of the source dataset directory (e.g., 'dc1000').
        combined_path (str): The path to the 'combined' directory where processed
                             files will be saved.
        dataset_name (str): The name of the dataset to prepend to filenames.
    """
    output_images_dir = os.path.join(combined_path, "images")
    output_labels_dir = os.path.join(combined_path, "labels")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    class_id = 0

    label_files_to_process = []
    for dataset_type in ["org_train_dataset", "org_test_dataset"]:
        labels_dir = os.path.join(base_path, dataset_type, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for label_filename in os.listdir(labels_dir):
            if label_filename.endswith(".png"):
                label_files_to_process.append(os.path.join(labels_dir, label_filename))

    for label_path in tqdm(label_files_to_process, desc=f"Processing {dataset_name}"):
        label_filename = os.path.basename(label_path)

        image_path = os.path.join(
            os.path.dirname(label_path), "..", "images", label_filename
        )

        if not os.path.exists(image_path):
            print(
                f"Warning: Corresponding image for {label_filename} not found. Skipping."
            )
            continue

        # Read original image for dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        img_height, img_width, _ = image.shape

        # Read the label mask in grayscale
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {label_path}. Skipping.")
            continue

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_labels = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Normalize coordinates for YOLO format
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            yolo_labels.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        # Create new prepended filenames
        new_image_filename = f"{dataset_name}_{label_filename}"
        new_label_filename = f"{dataset_name}_{os.path.splitext(label_filename)[0]}.txt"

        # Define final destination paths
        dest_image_path = os.path.join(output_images_dir, new_image_filename)
        dest_label_path = os.path.join(output_labels_dir, new_label_filename)

        # Copy the original image to the combined folder with the new name
        shutil.copy(image_path, dest_image_path)

        # Write the YOLO labels to a .txt file in the combined folder
        with open(dest_label_path, "w") as f:
            for line in yolo_labels:
                f.write(line + "\n")

    print(f"Process complete!")


if __name__ == "__main__":
    # The script is in 'datasets/scripts', so we navigate relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the source dc1000 dataset
    dc1000_path = os.path.join(current_dir, "..", "dc1000")

    # Path to the global combined dataset directory
    combined_dataset_path = os.path.join(current_dir, "..", "smile", "combined")

    process_and_integrate_dataset(
        dc1000_path, combined_dataset_path, dataset_name="dc1000"
    )
