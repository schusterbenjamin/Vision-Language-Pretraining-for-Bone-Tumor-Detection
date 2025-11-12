import os
import pydicom

count_mr = 0
count_ct = 0
count_cr = 0
count_other = 0

def move_if_x_ray(file_path):
    global count_mr, count_ct, count_other, count_cr
    try:
        ds = pydicom.dcmread(file_path)
        modality = ds.get('Modality', 'No Modality')
        print(f"File: {file_path}, Modality: {modality}")
        if modality == 'CR':
            # move image to ../no_tumour_only_xray folder
            print(f"Moving {file_path} to {os.path.join('../no_tumour_only_xray', stripped_path)}")
            # strip the first folder from the path
            parts = file_path.strip(os.sep).split(os.sep)
            stripped_path = os.sep.join(parts[1:])
            # os.copy(file_path, os.path.join('../no_tumour_only_xray', stripped_path))
            count_cr += 1
        else:
            # print(f"Skipping File: {file_path}, Not an X-Ray: Modality: {modality}")
            if modality == 'MR':
                count_mr += 1
            elif modality == 'CT':
                count_ct += 1
            else:
                count_other += 1
    except Exception as e:
        print(f"Error reading DICOM file: {e}")

if __name__ == "__main__":
    parent_path = '/mnt/nfs/homedirs/benjamins/DATA/no_tumour'
    for patient_folder in os.listdir(parent_path):
        patient_folder = os.path.join(parent_path, patient_folder)
        # check if the file is a folder
        if os.path.isdir(patient_folder):
            for scan_folder in os.listdir(patient_folder):
                scan_folder = os.path.join(patient_folder, scan_folder)
                # check if the file is a folder
                if os.path.isdir(scan_folder):
                    for file in os.listdir(scan_folder):
                        file_path = os.path.join(scan_folder, file)
                        # check if it is a png or db file
                        if file.endswith('.png') or file.endswith('.db'):
                            # print(f"Skipping File: {file_path}, since it is a png or db file")
                            continue

                        # assume the file is a dicom file
                        move_if_x_ray(file_path)


    print(f"Number of MR files: {count_mr}")
    print(f"Number of CT files: {count_ct}")
    print(f"Number of CR files: {count_cr}")
    print(f"Number of other files: {count_other}")
            
            

            
        
