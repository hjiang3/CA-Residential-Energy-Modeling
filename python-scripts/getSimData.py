import sys
import os 
import pandas as pd
import pdb # pdb.set_trace()
from matplotlib import pyplot as plt
import pickle

#%% === temp_eval func ===
def get_SimData(output_dir):
    """ custom function to evaluate overheating risk
        output_dir: output folder direcotries (absolute path)
        return a pd df to indicatre whether the temp criterion was met or not for each single building
    """
    df_temp = pd.DataFrame()
    df_cool = pd.DataFrame()

    count = 0
    bad_list = []
    for batch_dir in os.listdir(output_dir):
        # direcotry path of batch simulation results 
        batch_dir_path = os.path.join(output_dir, batch_dir)
        building_ID = int(batch_dir[4:])         
        try:
            # get time series results
            file_path = os.path.join(batch_dir_path, 'run', 'results_timeseries.csv')
            data = pd.read_csv(file_path)
            data = data.drop(0).reset_index(drop = True)
            ts = data['Time']
            if not 'Time' in df_temp.columns:
                df_temp.loc[:, 'Time'] = ts    
            if not 'time' in df_cool.columns:
                df_cool.loc[:, 'Time'] = ts
                
            # indoor temp
            temp = data.loc[:,'Temperature: Living Space'].astype(float)
            
            # AC energy use
            if 'End Use: Electricity: Cooling' in data.columns:
                cool_elec = data.loc[:, 'End Use: Electricity: Cooling'].astype(float)
            else:
                cool_elec = pd.DataFrame(columns = [building_ID])
            
            # overheating assessment
            # temp_thr = 82
            # time_thr = 0.04
            # count_overheat = temp > temp_thr
            # count_time = (count_overheat.sum()/len(temp)) > time_thr
            df_temp.loc[:, building_ID] = temp
            df_cool.loc[:, building_ID] = cool_elec  
        except:
            count += 1
            bad_list.append(building_ID)
                
    print(f'{count} buildings don\'t have simulation results')
    return bad_list, df_temp, df_cool
    
#%% apply temp_eval func to batch simulation results
output_dir = "E:/BatchEnergyModeling/ResStock/resstock-3.1.1/project_california/50k_batch/california_baseline_50k_modified"
bad_list, temp_df, cool_df = get_SimData(output_dir)

#%% save data
# Save the DataFrame to a pickle file
temp_df.to_pickle('C:/AB209/temp_df.pickle')
cool_df.to_pickle('C:/AB209/cool_df.pickle')

# Open the file in binary write mode and save the list
with open('D:/california_baseline_results/bad_list.pickle', 'wb') as f:
    pickle.dump(bad_list, f)
    
#%% save data
temp_df = pd.read_pickle('D:/california_baseline_results/overheat_df.pickle')
