import pandas as pd
import sys
import os
import shutil
import zipfile
import pdb # pdb.set_trace()
from datetime import datetime
sys.path.insert(0, os.path.join("C:/Users/yan85/Downloads/epw-master/epw-master/epw"))
from epw import epw

#%% custom functions
# rename and move EPW files
def copy_and_rename_files(org_wea_files, wea_filename, destination_folder, countyID):

    # List all files in the source folder
    files = os.listdir(org_wea_files)
    
    for file in files:
        if file == wea_filename:
            # Create the source file path
           source_file = os.path.join(org_wea_files, file)

           # Create the destination file path
           destination_file = os.path.join(destination_folder, countyID + '.epw')

           # Copy the file from source to destination
           shutil.copyfile(source_file, destination_file)
            
           print(f"File renamed and relocated: {file} -> {destination_folder}")

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
    
#%% change future EPW
# get ca_weather metadata
ca_weather_meta = pd.read_csv("D:/Yan/AB209/WeatherFiles/future weather 2050s/ca_weather_files_2050s.csv")

main_dir = "D:/Yan/AB209/WeatherFiles"
org_wea_files = os.path.join(main_dir, "future weather 2050s/downloads")

a = epw()
for index, row in ca_weather_meta.iterrows():
    # original future weather files folders
    epw_file = row['new_folder_name'] + '.epw'
    epw_path = os.path.join(main_dir, org_wea_files, epw_file)
    a.read(epw_path)
    df = a.dataframe
    df.iloc[:,-1] = 0
    df.iloc[:,-1]
    # correct wind speed values 
    wind = df['Wind Speed']
    df.loc[wind < 0, 'Wind Speed'] = float('nan')
    df['Wind Speed'].fillna(method='ffill', inplace=True)
    a.write(epw_path)

#%% replace TMY3 weather files with future weather files
# rename CA weather files and save in ToReplace folder
destination_folder_name = os.path.join(main_dir, "future weather 2050s/ToReplace")

for index, row in ca_weather_meta.iterrows():
    # original future weather files folders
    wea_filename = row['new_folder_name'] + '.epw'
    # countyID
    countyID = row['county_id']
    # rename future weather files with county IDs and move them to new folders
    copy_and_rename_files(org_wea_files, wea_filename, destination_folder_name, countyID)

#%% move renamed weather files to original zip folders
zipped_folder_path = os.path.join(main_dir, "BuildStock_Future.zip")
move_files_to_zip_folder(main_dir, destination_folder_name, zipped_folder_path)

#%% 
epw_file = '725946-24286-JACK MCNAMARA FIELD ARPT-1972-Cooling-n4-nooaminmaxT-realsky-realwind-realavgspechum-ThWilk-FutProj-HADCM3_A2A_2050-uhi-none-nasasun-nasawind.epw'
epw_path = os.path.join(main_dir, org_wea_files, epw_file)
a.read(epw_path)
df = a.dataframe
df.loc[4410: 4460, 'Wind Speed'].plot()

#%% get the hottest week in each EPW file
# get ca_weather metadata
ca_weather_meta = pd.read_csv("E:/Yan/AB209/WeatherFiles/future weather 2050s/ca_weather_files_2050s.csv")
org_wea_files = os.path.join("E:/Yan/AB209/WeatherFiles", "future weather 2050s/downloads")
a = epw()

hottestDay = []
for index, row in ca_weather_meta.iterrows():
    # original future weather files folders
    wea_filename = row['new_folder_name'] + '.epw'
    
    # countyID
    countyID = row['county_id']
    
    # read epw
    epw_path = os.path.join(org_wea_files, wea_filename)
    a.read(epw_path)
    df = a.dataframe
    
    # hottest day
    oat = df['Dry Bulb Temperature']
    hotest_idx = oat.idxmax()
    hottest_month = df['Month'][hotest_idx]
    hottest_day = df['Day'][hotest_idx]
    
    dt_object = datetime(2007, hottest_month, hottest_day)
    datetime_string = dt_object.strftime('%Y-%m-%d')
    hottestDay.append(datetime_string)

heatweek = pd.DataFrame({'hottestDay': hottestDay})
heatweek['county_id'] = ca_weather_meta['county_id']    

heatweek.to_pickle('heatweek.pkl')
    
