import os
import warnings
import cv2
import dicom2jpg
from tqdm import tqdm

dicom_dir = "~/DATA/healthy_subset_new/"
dicom_dir = os.path.expanduser(dicom_dir)

# sanity check: list all files in the directory
if not os.path.exists(dicom_dir):
    raise FileNotFoundError(f"The directory {dicom_dir} does not exist.")
if not os.path.isdir(dicom_dir):
    raise NotADirectoryError(f"The path {dicom_dir} is not a directory.")
# list all files in the directory
files = os.listdir(dicom_dir)
# only keep files that end with .dcm
files = [f for f in files if f.lower().endswith('.dcm')]

failed_conversion_files = []
warning_files = []
for file_name in tqdm(files, desc="Converting DICOM to PNG"):
    # check if it is a DICOM file
    if not file_name.lower().endswith('.dcm'):
        print(f"Skipping non-DICOM file: {file_name}")
        continue

    dicom_file_path = os.path.join(dicom_dir, file_name)
    
    # Construct the desired PNG filename and path
    # Remove the .dcm extension and add .png
    base_name = os.path.splitext(file_name)[0]
    output_png_path = os.path.join(dicom_dir, f"{base_name}.png")

    with warnings.catch_warnings(record=True) as w:
        # Ensure all warnings are caught
        warnings.simplefilter("default")
        try:
            img_data = dicom2jpg.dicom2img(dicom_file_path)
            cv2.imwrite(output_png_path, img_data)
        except Exception as e:
            # print(f"Failed to convert {dicom_file_path}: {e}")
            failed_conversion_files.append((dicom_file_path, e))

        # Check if a UserWarning was raised
        if w:
            if any(isinstance(warning.message, UserWarning) for warning in w):
                warning_files.append((dicom_file_path, [warning for warning in w if isinstance(warning.message, UserWarning)]))  
                print(warning_files)
                exit()

print(f"Encountered {len(warning_files)} warnings during conversion:")
for file_path, warns in warning_files:
    print(f"File: {file_path}, Exceptions: {[warning.message for warning in warns]}")

print(f"Failed to convert {len(failed_conversion_files)} files:")
for failed_file, exception in failed_conversion_files:
    print(f"File: {failed_file}, Exception: {exception}")

# print(result)