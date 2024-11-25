import pandas as pd
import os
import shutil
import zipfile

#%% custom functions
# rename and move EPW files
def copy_and_rename_files(main_dir, wea_folder, destination_folder, countyID):
    # open the zip file for reading
    with zipfile.ZipFile(wea_folder, 'r') as zip_ref:  
        # create a new folder for the extracted files
        new_folder_path = os.path.join(main_dir, destination_folder)
        # extract each file ending with '.epw', '.ddy' or '.stat'
        for file_info in zip_ref.infolist():
            # get file extension
            _, file_extension = os.path.splitext(file_info.filename)
            # find epw, ddy, stat
            if file_extension in ['.epw', '.ddy', '.stat']:
                zip_ref.extract(file_info, new_folder_path)
                # Rename the extracted file
                old_file_path = os.path.join(new_folder_path, file_info.filename)
                new_file_name = countyID + file_extension
                new_file_path = os.path.join(new_folder_path, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"File extracted and renamed: {file_info.filename} -> {new_file_name}")

# replace zipped EPW folders with new EPW files in an unzipped folder
def move_files_to_zip_folder(main_dir, unzipped_folder_path, zip_file_path):
    # create a temporary folder to hold the files from the zip file
    temp_folder = os.path.join(main_dir, "temp_folder")
    os.makedirs(temp_folder, exist_ok=True)

    # extract the contents of the zip file to the temporary folder
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)

    # move files from the unzipped folder to the temporary folder, replacing existing files
    for filename in os.listdir(unzipped_folder_path):
        src_file = os.path.join(unzipped_folder_path, filename)
        dest_file = os.path.join(temp_folder, filename)
        shutil.move(src_file, dest_file)

    # re-zip the contents of the temporary folder and overwrite the original zip file
    with zipfile.ZipFile(zip_file_path, 'w') as zip_ref:
        for foldername, subfolders, filenames in os.walk(temp_folder):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zip_ref.write(file_path, os.path.relpath(file_path, temp_folder))

    # clean up the temporary folder
    shutil.rmtree(temp_folder)

#%% replace TMY3 weather files with TMYx_2007-2021
# get ca_weather metadata
ca_weather_meta = pd.read_csv("E:/Yan/AB209/WeatherFiles/ca_weather_files.csv")

main_dir = "E:/Yan/AB209/WeatherFiles"
org_wea_files = os.path.join(main_dir, "WeatherFilesTMY2007-2021/downloads")
destination_folder_name = os.path.join(main_dir, "WeatherFilesTMY2007-2021/ToReplace")

# rename CA weather files and save in ToReplace folder
for index, row in ca_weather_meta.iterrows():
    # original TMY folders
    wea_folder = os.path.join(org_wea_files, row['new_folder_name'] + '.zip')
    # countyID
    countyID = row['county_id']
    # rename TMY files with county IDs and move them to new folders
    copy_and_rename_files(main_dir, wea_folder, destination_folder_name, countyID)
    
#%% move renamed weather files to original zip folders
zipped_folder_path = os.path.join(main_dir, "BuildStock_TMY3_FIPS.zip")
move_files_to_zip_folder(main_dir, destination_folder_name, zipped_folder_path)
