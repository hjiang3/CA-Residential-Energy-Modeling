""" Use the costum module for AB209 postprocessing
    @author: yan.wang@berkeley.edu
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

#%% === map sim results to CA counties ===
def map_plot(df, title, 
             overheating = 'No', 
             counties_shapefile_path = r"map/ca-county-boundaries/CA_Counties/CA_Counties_TIGER2016.shp", 
             color_schm = None, scal_dn = 0, scal_up = 1, min_max = None ):
    '''input a pd df: including 'County' and 'Value' columns
    '''
    vmin = round(min(df['Value']),1)
    vmax = round(max(df['Value']),1)
    
    # Merge the DataFrame with the California counties shapefile
    california_counties = gpd.read_file(counties_shapefile_path)
    california_counties = california_counties.merge(df, how='left', left_on='NAME', right_on='County')
    
    # Plot the map
    fig, ax = plt.subplots(1, figsize=(9, 6))
    plt.axis('off')
    tick_cbar = np.arange(vmin, vmax + 0.1, 0.1)
    
    # different color bar scales for overheating plots
    if overheating == 'Yes' or  min_max != None:
        vmin = 0
        vmax = 1
        tick_cbar = np.arange(0, 1.1, 0.2)

    if color_schm == None:
        cmap='OrRd'
    else:
        cmap_modified = plt.colormaps.get_cmap(color_schm)
        cmap = ListedColormap(cmap_modified(np.linspace(scal_dn, scal_up, 256)))
        
    california_counties.plot(column='Value', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))   
    sm._A = []
    cbar = fig.colorbar(sm, cax=ax.inset_axes([1.05, 0.1, 0.04, 0.8]), ticks = tick_cbar)
    cbar.ax.set_yticklabels([f'{100*i:.0f}%'for i in tick_cbar])
    cbar.ax.tick_params(labelsize = 12)
    # ax.set_title(title, fontsize = 14)
    plt.show()
    fig_dir = title + '.jpg'
    fig.savefig(fig_dir, format='jpeg', pil_kwargs={'quality': 95})  

#%% === map overheating metrics ===
def overheating_eval(metadata, title, ts,
                     temp_df, cooling_df, cooling_allAC_df, 
                     threshold_temp, threshold_hour, threshold_day, 
                     ac_type = 'NoAC'):
    # 8 hour running mean
    n_roll = 8
    runmean_df = temp_df.rolling(n_roll).mean()
    runmean_df.iloc[:n_roll, :] = temp_df.iloc[:n_roll, :]

    # Group by date and check if any hourly value exceeds the threshold_temp
    runmean_df['Date'] = [pd.Timestamp(time_str).strftime('%Y/%m/%d') for time_str in ts]
    overheat = runmean_df.groupby(runmean_df['Date']).apply(lambda x: (x > threshold_temp).sum() > threshold_hour).sum() > threshold_day
    overheat_df = pd.DataFrame({'Building': overheat.index, 'Overheating': overheat})
    
    # df merge for map plot
    merged_df = metadata.merge(overheat_df, how ='left', on ='Building')
    
    if ac_type == 'NoAC':
        cond_AC = merged_df['HVAC Cooling Type'].isnull()

        # Baseline cooling energy use
        cooling_peak = max(cooling_df.sum(axis = 1))
        
        # no-AC buildings that don't meet the requirement
        cond_overheat = merged_df['Overheating'] == True
        col_overheat = merged_df.loc[cond_overheat & cond_AC, 'Building'].values.tolist()

        # pick buidling IDs from the all-AC df based on overheating results
        noAC_cols = [col for col in cooling_allAC_df.columns if col in col_overheat]
        percentage_noAC = round(len(noAC_cols) / sum(cond_AC) * 100, 1)
        print(f'{percentage_noAC}% no-AC buildings need to add ACs')
        
        # form new cooling df
        cooling_df_new = cooling_df.join(cooling_allAC_df[noAC_cols])    
        allAC_cooling_peak = max(cooling_df_new.sum(axis = 1))
        added_cooling_peak = allAC_cooling_peak - cooling_peak
        
        print(f'Added cooling peak load is estimated to be {round(added_cooling_peak/1000 * (12.2) * 10**6 / 52218)} MW (+{round(added_cooling_peak/cooling_peak * 100)}%) for CA')

        # subset for no-AC buildings
        merged_df = merged_df.loc[cond_AC, :]
    
    overheat_pct = merged_df.groupby(merged_df['County'])['Overheating'].apply(np.mean)
    overheat_county = pd.DataFrame({'County': overheat_pct.index, 'Value': overheat_pct})
    overheat_county.reset_index(drop = True, inplace = True)
    
    # plot
    map_plot(overheat_county, title, overheating = 'Yes')

#%% === map overheating metrics ===
def add_AC(metadata, title, ts,
            temp_df, temp_allAC_df, 
            cooling_df, cooling_allAC_df, 
            threshold_temp, threshold_hour, threshold_day):
            
    # 8 hour running mean
    n_roll = 8
    runmean_df = temp_df.rolling(n_roll).mean()
    runmean_df.iloc[:n_roll, :] = temp_df.iloc[:n_roll, :]

    # Group by date and check if any hourly value exceeds the threshold_temp
    runmean_df['Date'] = [pd.Timestamp(time_str).strftime('%Y/%m/%d') for time_str in ts]
    overheat = runmean_df.groupby(runmean_df['Date']).apply(lambda x: (x > threshold_temp).sum() > threshold_hour).sum() > threshold_day
    overheat_df = pd.DataFrame({'Building': overheat.index, 'Overheating': overheat})
    
    # df merge for map plot
    merged_df = metadata.merge(overheat_df, how ='left', on ='Building')
    
    cond_AC = merged_df['HVAC Cooling Type'].isnull()

    # Baseline cooling energy use
    cooling_peak = max(cooling_df.sum(axis = 1))
    cooling_tot = cooling_df.sum().sum()

    # no-AC buildings that don't meet the requirement
    cond_overheat = merged_df['Overheating'] == True
    col_overheat = merged_df.loc[cond_overheat & cond_AC, 'Building'].values.tolist()

    # pick buidling IDs from the all-AC df based on overheating results
    noAC_cols = [col for col in cooling_allAC_df.columns if col in col_overheat]
    print(f'{round(len(noAC_cols)/ sum(cond_AC) * 100, 1)}% no-AC buildings need to add ACs')
    
    # form new cooling df
    cooling_df_new = cooling_df.join(cooling_allAC_df[noAC_cols])  
    
    # added peak load
    allAC_cooling_peak = max(cooling_df_new.sum(axis = 1))
    added_cooling_peak = allAC_cooling_peak - cooling_peak

    # added total energy use 
    allAC_cooling_tot = cooling_df_new.sum().sum()
    added_cooling_tot = allAC_cooling_tot - cooling_tot

    print(f'Added cooling peak load is estimated to be {round(added_cooling_peak/1000 * (14.3) * 10**6 / 52218)} MW (+{round(added_cooling_peak/cooling_peak * 100)}%) for CA')

    print(f'Added cooling total energy is estimated to be {round(added_cooling_tot/1000 * (14.3) * 10**6 / 52218)} MWh (+{round(added_cooling_tot/cooling_tot * 100)}%) for CA')

    # form new cooling df
    temp_df_new = temp_df.copy()
    temp_df_new[noAC_cols] = temp_allAC_df[noAC_cols]
    runmean_df = temp_df_new.rolling(n_roll).mean()
    runmean_df.iloc[:n_roll, :] = temp_df.iloc[:n_roll, :]

    # Group by date and check if any hourly value exceeds the threshold_temp
    runmean_df['Date'] = [pd.Timestamp(time_str).strftime('%Y/%m/%d') for time_str in ts]
    overheat = runmean_df.groupby(runmean_df['Date']).apply(lambda x: (x > threshold_temp).sum() > threshold_hour).sum() > threshold_day
    overheat_df = pd.DataFrame({'Building': overheat.index, 'Overheating': overheat})
    
    merged_df = metadata.merge(overheat_df, how ='left', on ='Building')
    overheat_pct = merged_df.groupby(merged_df['County'])['Overheating'].apply(mean_exclude_top5_percent)
    # overheat_pct['CA, Trinity County'] = overheat_pct['CA, Tehama County']
    overheat_county = pd.DataFrame({'County': overheat_pct.index, 'Value': overheat_pct})
    overheat_county.reset_index(drop = True, inplace = True)
    
    # save to csv
    tab_dir = os.path.join('results/tables', title + '.csv')
    overheat_county.to_csv(tab_dir, index = False)

#%% Function to calculate mean excluding the top 5% of values within each group
def mean_exclude_top5_percent(group):
    # Determine the cutoff index for the top 5%
    cutoff_index = int(len(group) * 0.95)
    # Sort the values and exclude the top 5%
    sorted_group = group.sort_values().iloc[:cutoff_index]
    # Return the mean of the remaining values
    return sorted_group.mean()

#%% save overheating hours
def overheating_hours_tab(metadata, title, temp_df, threshold_temp, 
                            ac_type = None, deg_hours = None, avg = 'mean', heatwave = None, scal = 'low'):
    # 8 hour running mean
    n_roll = 8
    runmean_df = temp_df.rolling(n_roll).mean()
    runmean_df.iloc[:n_roll, :] = temp_df.iloc[:n_roll, :]

    # if hourly temp exceeds the threshold_temp
    exceed_hours = (runmean_df > threshold_temp).sum(axis = 0)
    # heatwave week
    if heatwave == 'Yes':
        exceed_hours = exceed_hours
    else:
        exceed_hours = exceed_hours

    if deg_hours == 'Yes':
        cond = (runmean_df > threshold_temp).astype(int)
        deg = runmean_df - threshold_temp
        deg[deg < 0] = 0
        exceed_hours = (cond * deg).sum(axis = 0)
    
    overheat_df = pd.DataFrame({'Building': exceed_hours.index, 'exceed_hours': exceed_hours}).reset_index(drop = True)

    # df merge for map plot
    merged_df = metadata.merge(overheat_df, how ='left', on ='Building') 
    
    if ac_type == 'AC':
        cond_AC = merged_df['HVAC Cooling Type'].notnull()
        merged_df = merged_df.loc[cond_AC, :]
    elif ac_type == 'noAC':
        cond_AC = merged_df['HVAC Cooling Type'].isnull()
        merged_df = merged_df.loc[cond_AC, :]

    if avg == 'mean':
        overheat_pct = merged_df.groupby(merged_df['County'])['exceed_hours'].apply(mean_exclude_top5_percent)
    elif avg == 'median':
         overheat_pct = merged_df.groupby(merged_df['County'])['exceed_hours'].median()
    
    overheat_pct['Trinity'] = overheat_pct['Tehama']

    overheat_county = pd.DataFrame({'County': overheat_pct.index, 'Value': overheat_pct})
    overheat_county.reset_index(drop = True, inplace = True)
    
    # save to csv
    tab_dir = os.path.join('results/tables', title + '.csv')
    overheat_county.to_csv(tab_dir, index = False)

#%% === map overheating hours ===
def cool_energy_county(metadata, title, cool_df, ac_type = None, avg = 'mean'):
        
    # total cooling energy use
    cool_energy_df = pd.DataFrame({'Building': cool_df.columns, 'cool_energy': cool_df.sum(axis = 0)}).reset_index(drop = True)

    # df merge for map plot
    merged_df = metadata.merge(cool_energy_df, how ='left', on ='Building') 
    
    # AC homes
    if ac_type == 'AC':
        cond_AC = merged_df['HVAC Cooling Type'].notnull()
        merged_df = merged_df.loc[cond_AC, :]

    if avg == 'mean':
        ac_county = merged_df.groupby(merged_df['County'])['cool_energy'].apply(mean_exclude_top5_percent)
    elif avg == 'median':
        ac_county = merged_df.groupby(merged_df['County'])['cool_energy'].median()
    
    # correction 
    ac_county['Trinity'] = ac_county['Tehama']

    output_df = pd.DataFrame({'County': ac_county.index, 'Avg_Tot_AC_Use(KWh)': ac_county})
    output_df.reset_index(drop = True, inplace = True)
    
    # save to csv
    tab_dir = os.path.join('results/tables', title + '_cool_energy.csv')
    output_df.to_csv(tab_dir, index = False)

#%% === map overheating hours ===
def overheating_hours(metadata, title, temp_df, threshold_temp, 
                      counties_shapefile_path = r"C:/Users/map/ca-county-boundaries/CA_Counties/CA_Counties_TIGER2016.shp", 
                      ac_type = None, deg_hours = None, avg = 'mean', heatwave = None, scal = 'low'):
    # 8 hour running mean
    n_roll = 8
    runmean_df = temp_df.rolling(n_roll).mean()
    runmean_df.iloc[:n_roll, :] = temp_df.iloc[:n_roll, :]

    # if hourly temp exceeds the threshold_temp
    exceed_hours = (runmean_df > threshold_temp).sum(axis = 0)
    # heatwave week
    if heatwave == 'Yes':
        exceed_hours = exceed_hours/168
    else:
        exceed_hours = exceed_hours/4416

    if deg_hours == 'Yes':
        cond = (runmean_df > threshold_temp).astype(int)
        deg = runmean_df - threshold_temp
        deg[deg < 0] = 0
        exceed_hours = (cond * deg).sum(axis = 0)
    
    overheat_df = pd.DataFrame({'Building': exceed_hours.index, 'exceed_hours': exceed_hours}).reset_index(drop = True)

    # df merge for map plot
    merged_df = metadata.merge(overheat_df, how ='left', on ='Building') 
    
    if ac_type == 'AC':
        cond_AC = merged_df['HVAC Cooling Type'].notnull()
        merged_df = merged_df.loc[cond_AC, :]
    elif ac_type == 'noAC':
        cond_AC = merged_df['HVAC Cooling Type'].isnull()
        merged_df = merged_df.loc[cond_AC, :]

    if avg == 'mean':
        overheat_pct = merged_df.groupby(merged_df['County'])['exceed_hours'].apply(mean_exclude_top5_percent)
    elif avg == 'median':
        overheat_pct = merged_df.groupby(merged_df['County'])['exceed_hours'].median()
    
    overheat_pct['Trinity'] = overheat_pct['Tehama']

    overheat_county = pd.DataFrame({'County': overheat_pct.index, 'Value': overheat_pct})
    overheat_county.reset_index(drop = True, inplace = True)
    
    # save to csv
    # tab_dir = os.path.join('results/tables', title + '.csv')
    # overheat_county.to_csv(tab_dir, index = False)
    
    # plot
    if heatwave == None:
        if max(overheat_pct) < 100:
            vmin = round(min(overheat_pct)/10) * 10
            vmax = round(max(overheat_pct)/10) * 10
        else:
            vmin = round(min(overheat_pct)/100) * 100
            vmax = round(max(overheat_pct)/100) * 100
        
        vmax = 0.08
        vmin = 0
        
        if scal == 'high':
            vmax = 1
            cmap_step = 0.2
            
        elif scal == 'med':
            vmax = 0.6
            cmap_step = 0.15   
    else:
        vmin = 0
        if scal == 'high':
            vmax = 1
            cmap_step = 0.2
        elif scal == 'med':
            vmax = 0.6
            cmap_step = 0.15    
        else:
            vmax = 0.2
            cmap_step = 0.05
    
    # Merge the DataFrame with the California counties shapefile
    california_counties = gpd.read_file(counties_shapefile_path)
    california_counties = california_counties.merge(overheat_county, how='left', left_on='NAME', right_on='County')
    
    # Plot the map
    fig, ax = plt.subplots(1)
    plt.axis('off')

    if heatwave == 'Yes':
        tick_cbar = np.arange(0, 1.05, cmap_step)
    else:    
        # tick_cbar = np.arange(vmin, vmax+1, 50)
        tick_cbar = np.arange(0, vmax+0.01, 0.02)
        if scal == 'high':
            tick_cbar = np.arange(0, vmax+0.01, 0.2)

    california_counties.plot(column='Value', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax)    
    
    # Calculate the aspect ratio
    original_width, original_height = fig.get_figure().get_size_inches()
    scal_fac = 1.2
    california_counties.plot(column = 'Value', cmap = 'OrRd', linewidth = 0.8, ax = ax, edgecolor = '0.8', 
                                             vmin = vmin, vmax = vmax)    
    
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=ax.inset_axes([1.05, 0.1, 0.04, 0.8]), ticks = tick_cbar)

    # if heatwave == 'Yes':
    cbar.ax.set_yticklabels([f'{100*i:.0f}%'for i in tick_cbar])
    cbar.ax.set_ylim(0,vmax)

    cbar.ax.tick_params(labelsize = 12)
    ax.set_title(title, fontsize = 12)
    fig.set_size_inches(original_width*scal_fac, original_height*scal_fac)
    plt.show()
    fig.savefig(title + '.jpg', format='jpeg', pil_kwargs={'quality': 95})

#%% === get overheating hours statistics ===
def overheating_hours_stat(metadata, title, temp_df, threshold_temp, ac_type = None, deg_hours = None):
    # 8 hour running mean
    n_roll = 8
    runmean_df = temp_df.rolling(n_roll).mean()
    runmean_df.iloc[:n_roll, :] = temp_df.iloc[:n_roll, :]

    # if hourly temp exceeds the threshold_temp
    exceed_hours = (runmean_df > threshold_temp).sum(axis = 0)
    
    if deg_hours == 'Yes':
        cond = (runmean_df > threshold_temp).astype(int)
        deg = runmean_df - threshold_temp
        deg[deg < 0] = 0
        exceed_hours = (cond * deg).sum(axis = 0)
    
    overheat_df = pd.DataFrame({'Building': exceed_hours.index, 'exceed_hours': exceed_hours}).reset_index(drop = True)
    
    # df merge for map plot
    merged_df = metadata.merge(overheat_df, how ='left', on ='Building') 
    
    if ac_type == 'AC':
        cond_AC = merged_df['HVAC Cooling Type'].notnull()
        merged_df = merged_df.loc[cond_AC, :]
    elif ac_type == 'noAC':
        cond_AC = merged_df['HVAC Cooling Type'].isnull()
        merged_df = merged_df.loc[cond_AC, :]
    
    grouped = merged_df.groupby(merged_df['County'])['exceed_hours']
    # Calculate statistics for each group
    statistics = {}
    for group_name, group_data in grouped:
        statistics[group_name] = {
            'Mean': group_data.mean(),
            '5% percentile': group_data.quantile(0.05),
            '25% percentile': group_data.quantile(0.25),
            'Median': group_data.median(),
            '75% percentile': group_data.quantile(0.75),
            '95% percentile':group_data.quantile(0.95),
            'Min': group_data.min(),
            'Max': group_data.max(),
            }
    
    # Display statistics for each group
    for group_name, stats in statistics.items():
        print(f"Group '{group_name}':")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")