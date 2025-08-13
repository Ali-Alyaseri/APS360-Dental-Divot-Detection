import os
import shutil
import random
import logging
from tqdm import tqdm


def setup_logging(log_file_path):
    """Configures logging to write to a file."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ],
    )


def create_split_datasets(
    base_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
):
    """
    Splits a combined dataset into training, validation, and test sets and logs the results.
    """

    log_file_path = os.path.join(output_dir, "split_log.txt")
    setup_logging(log_file_path)

    source_images_dir = os.path.join(base_dir, "images")
    source_labels_dir = os.path.join(base_dir, "labels")

    all_filenames = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(source_images_dir)]
    )
    random.seed(42)  # Use a fixed seed for reproducibility
    random.shuffle(all_filenames)

    # Calculate split indices
    total_files = len(all_filenames)
    train_end = int(total_files * train_ratio)
    val_end = int(total_files * (train_ratio + val_ratio))

    # Create splits
    train_files = all_filenames[:train_end]
    val_files = all_filenames[train_end:val_end]
    test_files = all_filenames[val_end:]

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split_name, filenames in splits.items():

        dest_img_dir = os.path.join(output_dir, split_name, "images")
        dest_lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)

        for filename in tqdm(filenames, desc=f"Copying {split_name} files"):
            # Determine image extension (.jpg or .png)
            img_ext = (
                ".jpg"
                if os.path.exists(os.path.join(source_images_dir, filename + ".jpg"))
                else ".png"
            )

            src_img_path = os.path.join(source_images_dir, filename + img_ext)
            src_lbl_path = os.path.join(source_labels_dir, filename + ".txt")

            if not os.path.exists(src_img_path):
                logging.warning(
                    f"Could not find image for base name {filename}. Skipping."
                )
                continue

            shutil.copy(src_img_path, os.path.join(dest_img_dir, filename + img_ext))
            if os.path.exists(src_lbl_path):
                shutil.copy(src_lbl_path, os.path.join(dest_lbl_dir, filename + ".txt"))

    logging.info(f"Total files processed: {total_files}")
    logging.info(
        f"Training set size:   {len(train_files)} files ({len(train_files)/total_files:.2%})"
    )
    logging.info(
        f"Validation set size: {len(val_files)} files ({len(val_files)/total_files:.2%})"
    )
    logging.info(
        f"Test set size:       {len(test_files)} files ({len(test_files)/total_files:.2%})"
    )


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    combined_dir = os.path.join(current_dir, "..", "smile", "combined")
    split_dir = os.path.join(current_dir, "..", "smile", "split2")

    create_split_datasets(combined_dir, split_dir)
