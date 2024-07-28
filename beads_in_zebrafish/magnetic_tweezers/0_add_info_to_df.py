'''
Ursa Ursic, last updated: 26.3.2024

This file takes the data from "magnetic tweezers" folder and creates csv files with additional information. 
Additional info:
- distance from tip
- force
- magnet pulses info
- MT info 

It also saves the tip images with tip outlines. 
You only need to run this file once. The rest of the analysis follows in a different file. 
'''

import pandas as pd
import os
from tqdm import tqdm

from utils import *

################################################################################
######################### DEFINE DIRECTORIES ###################################
# Connect to cytokinesis-zebrafish-collab server from your computer
filepath_measurements_info = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers/2_analysis/CC_phases_tweezers_info.csv'

# Define the result directory
dir_plots = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers/3_plots'
dir_analysis = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers/2_analysis'

# Folder where th extended measurements are going to be saved
dir_measurements_extended = f'{dir_analysis}/measurements_extended_info'
################################################################################


def main():
    # General info dataframe
    df_general_info = pd.read_csv(filepath_measurements_info, delimiter=';', encoding='utf-8')

    if not os.path.exists(dir_measurements_extended):
        os.mkdir(dir_measurements_extended)
    
    for idx in tqdm(range(len(df_general_info))):
        # Load data
        filepath = df_general_info['trackmate_file'].values[idx]
        filename = os.path.basename(filepath).split('.')[0]  # This is the name of the file without extention
        df = pd.read_csv(filepath, skiprows=[1, 2, 3], encoding = "utf-8")

        # Find tip, save tip img to file and calculate distance form tip
        filepath_tip = df_general_info['before_file'].values[idx]
        threshold_tip = df_general_info['tip_threshold'].values[idx]
        dir_tip_imgs = f'{dir_plots}/tip_images'
        if not os.path.exists(dir_tip_imgs):
            os.mkdir(dir_tip_imgs)
        filepath_tip_outline = f'{dir_tip_imgs}/{filename}.png'
        tip = find_tip(filepath_tip, threshold_tip, save_img_to_path=filepath_tip_outline, endpoint=True)
        if tip:
            calculate_distance_from_tip(df, tip) 
        else:
            continue

        # Add additional information about magnet status 
        magnet_info = df_general_info[['first_pulse (frame)', 't_on (frame)', 't_off (frame)']].values[idx]
        add_magnet_status(df, magnet_info)

        # Calculate force on beads
        calibration = df_general_info['calibration (mV)'].values[idx]
        calculate_force(df, calibration)

        # add information about MTs 
        MT_info = df_general_info[['m_phase_1_start (frame)', 'm_phase_1_end (frame)', 
                                    'i_phase_1_start (frame)', 'i_phase_1_end (frame)', 
                                    'm_phase_2_start (frame)', 'm_phase_2_end (frame)', 
                                    'i_phase_2_start', 'i_phase_2_end',]].values[idx]
        add_MT_status(df, MT_info)
        df.attrs['COMMENTS'] = str(df_general_info["comments"].values[idx])

        # save the extended df to a file (in analysis folder)
        filepath_extended_df = f'{dir_measurements_extended}/{filename}_extended.h5'
        df.to_hdf(filepath_extended_df, key='df', mode='w', format='table')
        # Save DataFrame to HDF5 file with comments
        with pd.HDFStore(filepath_extended_df, mode='w') as store:
            store.put('df', df)
            store.get_storer('df').attrs.metadata = df.attrs['COMMENTS']


if __name__ == '__main__':
    main()