import pandas as pd
import os

# --- Part 1: Filter the CSV file ---

# This part is the same as before, creating the filtered CSV.
# It is kept hardcoded as requested.
df = pd.read_csv('meta/bounding_boxes.csv')
for col in ['image_width', 'image_height', 'bbox_width', 'bbox_height']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=['image_width', 'image_height', 'bbox_width', 'bbox_height'], inplace=True)
image_area = df['image_width'] * df['image_height']
bbox_area = df['bbox_width'] * df['bbox_height']
df['relative_area_percent'] = (bbox_area / image_area) * 100
mask_to_keep = (df['relative_area_percent'] >= 0.1) & \
               (df['image_filename'].str.startswith('dc1000_'))
filtered_df = df[mask_to_keep]
original_columns = ['image_filename', 'image_width', 'image_height', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max', 'bbox_width', 'bbox_height']
columns_to_save = [col for col in original_columns if col in df.columns]
filtered_df.to_csv('filtered_bounding_boxes.csv', index=False)


# --- Part 2: Clean up the image and label directories ---

print("\n--- Starting Directory Cleanup ---")

# Define the paths to your data directories
images_dir = 'images'
labels_dir = 'labels'
filtered_csv_path = 'filtered_bounding_boxes.csv'
filename_prefix = 'dc1000_'

# Check if the necessary directories and file exist
if not os.path.isdir(images_dir):
    print(f"Error: The directory '{images_dir}' was not found.")
    exit()
if not os.path.isdir(labels_dir):
    print(f"Error: The directory '{labels_dir}' was not found.")
    exit()
if not os.path.exists(filtered_csv_path):
    print(f"Error: The filtered CSV '{filtered_csv_path}' was not found. Make sure Part 1 ran correctly.")
    exit()

# Read the filtered CSV to get a list of files to keep
try:
    files_to_keep_df = pd.read_csv(filtered_csv_path)
    # Create a unique set of filenames for fast lookups
    files_to_keep_set = set(files_to_keep_df['image_filename'].unique())
    print(f"Found {len(files_to_keep_set)} unique image files to keep.")
except Exception as e:
    print(f"Error reading {filtered_csv_path}: {e}")
    exit()


# Iterate over all files in the images directory
print(f"Scanning '{images_dir}' for files to delete...")
for filename in os.listdir(images_dir):
    # IMPORTANT: Only consider files that start with the specified prefix
    if filename.startswith(filename_prefix):
        # Check if the current image filename is in our set of files to keep
        if filename not in files_to_keep_set:
            # This file is not in the filtered CSV, so we should delete it.

            # 1. Delete the image file
            image_path_to_delete = os.path.join(images_dir, filename)
            try:
                os.remove(image_path_to_delete)
                print(f"Deleted image: {image_path_to_delete}")
            except OSError as e:
                print(f"Error deleting image {image_path_to_delete}: {e}")

            # 2. Delete the corresponding label file
            base_filename, _ = os.path.splitext(filename)
            label_filename = f"{base_filename}.txt"
            label_path_to_delete = os.path.join(labels_dir, label_filename)

            try:
                os.remove(label_path_to_delete)
                print(f"Deleted label: {label_path_to_delete}")
            except FileNotFoundError:
                print(f"Info: No corresponding label file found at {label_path_to_delete}")
            except OSError as e:
                print(f"Error deleting label {label_path_to_delete}: {e}")

print("\n--- Directory Cleanup Complete ---")


# --- Part 3: Synchronize individual label files with the filtered CSV ---

print("\n--- Starting Label File Synchronization ---")

# Group the filtered dataframe by filename for efficient lookup
grouped_filtered_df = filtered_df.groupby('image_filename')

for image_filename, group in grouped_filtered_df:
    base_filename, _ = os.path.splitext(image_filename)
    label_filename = f"{base_filename}.txt"
    label_filepath = os.path.join(labels_dir, label_filename)

    if not os.path.exists(label_filepath):
        continue

    # Generate the set of valid YOLO strings from the filtered CSV data for this image
    valid_yolo_lines = set()
    for _, row in group.iterrows():
        # Calculate YOLO format values
        x_center = (row['x_min'] + (row['bbox_width'] / 2)) / row['image_width']
        y_center = (row['y_min'] + (row['bbox_height'] / 2)) / row['image_height']
        width = row['bbox_width'] / row['image_width']
        height = row['bbox_height'] / row['image_height']

        # Create the YOLO string. Format to 6 decimal places like standard YOLO.
        yolo_line = f"{int(row['class_id'])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        valid_yolo_lines.add(yolo_line)

    # Read the existing label file and filter its lines
    try:
        with open(label_filepath, 'r') as f:
            lines_in_file = f.readlines()

        # Strip whitespace and filter
        lines_in_file_stripped = [line.strip() for line in lines_in_file]

        # Find which lines from the file are actually valid
        final_lines_to_keep = []
        for line in lines_in_file_stripped:
            if line in valid_yolo_lines:
                final_lines_to_keep.append(line)

        # Compare original and filtered line counts to see if a change was made
        if len(final_lines_to_keep) != len(lines_in_file_stripped):
            print(f"Updating label file: {label_filepath}. Removed {len(lines_in_file_stripped) - len(final_lines_to_keep)} extra entries.")
            # Rewrite the file with only the valid lines
            with open(label_filepath, 'w') as f:
                for line in final_lines_to_keep:
                    f.write(f"{line}\n")

    except Exception as e:
        print(f"Error processing label file {label_filepath}: {e}")


print("\n--- Synchronization Complete ---")
