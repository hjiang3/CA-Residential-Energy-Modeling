import os
import pandas as pd
import pdb # pdb.set_trace()
from matplotlib import pyplot as plt

#%% setup 
output_dir = "D:\AB209SimOutputs\Baseline"
metadata_dir = os.path.join(output_dir, "california_baseline_50k_modified.csv")
metadata = pd.read_csv(metadata_dir)

temp_dir = os.path.join(output_dir, "temp_df.pickle")
temp_df = pd.read_pickle(temp_dir)

#%% get 5% and 95% percentiles
pcrt = temp_df.describe(percentiles=[0.05, 0.5, 0.95])

pcrt5 = pcrt.loc['5%']
pcrt95 = pcrt.loc['95%']

#%%  5% plot
plt.hist(pcrt5, bins= 30, density=True, alpha=0.5 )
plt.title('Histogram of 5% percentile of indoor air temperature in ResStock simulations')
plt.xlabel('Indoor air temperature (F)')
plt.ylabel('Frequency')

#%%  95% plot
plt.hist(pcrt95, bins= 30, density=True, alpha=0.5 )
plt.title('Histogram of 95% percentile of indoor air temperature in ResStock simulations')
plt.xlabel('Indoor air temperature (F)')
plt.ylabel('Frequency')


#%% CEC climate AC

metadata['HVAC Cooling Type'] = metadata['HVAC Cooling Type'].fillna('NoAC')
grouped = metadata.groupby(['CEC Climate Zone', 'HVAC Cooling Type']).size().unstack(fill_value=0)
normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
normalized_grouped.to_csv('D:\AB209SimOutputs\Baseline\CEC Climate Zone AC Saturation.csv')
# Plotting the histogram
normalized_grouped.plot(kind='barh', stacked=True)

