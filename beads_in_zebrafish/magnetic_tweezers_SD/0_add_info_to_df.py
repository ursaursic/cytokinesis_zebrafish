'''
Ursa Ursic, last updated: 28.7.2024

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
import argparse
import yaml

from utils import *


def main(config_path):
    # load config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    filepath_measurements_info = config['filepath_measurements_info']
    dir_plots = config['dir_parent']+'/3_plots'
    dir_analysis = config['dir_parent']+'/2_analysis'
    recalculate = config['recalculate']

    # Folder where the extended measurements are going to be saved
    dir_measurements_extended = f'{dir_analysis}/measurements_extended_info'

    # General info dataframe
    df_general_info = pd.read_csv(filepath_measurements_info, delimiter=';', encoding='utf-8')

    if not os.path.exists(dir_measurements_extended) :
        os.mkdir(dir_measurements_extended)
    
    for idx in tqdm(range(len(df_general_info))):
        # try:
        # Load data
        filepath = df_general_info['trackmate_file'].values[idx]
        filename = os.path.basename(filepath).split('.')[0]  # This is the name of the file without extention
        
        # define the data directory to which we save extended measurements
        filepath_extended_df = f'{dir_measurements_extended}/{filename}_extended.h5'

        if os.path.exists(filepath_extended_df) and not recalculate:
            continue


        df = pd.read_csv(filepath, skiprows=[1, 2, 3], encoding = "utf-8")
        # check if we have before file or end potint of the tip

        if 'before_file' in df_general_info.columns:
            # Find tip, save tip img to file and calculate distance form tip
            filepath_tip = df_general_info['before_file'].values[idx]
            threshold_tip = df_general_info['tip_threshold'].values[idx]
            dir_tip_imgs = f'{dir_plots}/tip_images'
            if not os.path.exists(dir_tip_imgs):
                os.mkdir(dir_tip_imgs)
            filepath_tip_outline = f'{dir_tip_imgs}/{filename}.png'

            tip = find_tip(filepath_tip, threshold_tip, save_img_to_path=filepath_tip_outline, endpoint=True)

        elif 'tip_x' in df_general_info.columns:
            tip = list(df_general_info[['tip_x', 'tip_y']].values[idx])

        if tip:
            calculate_distance_from_tip(df, tip) 
        else:
            continue

        # Add additional information about magnet status 
        magnet_info = df_general_info[['first_pulse (frame)', 't_on (frame)', 't_off (frame)']].values[idx]
        # if any(magnet_info == 'unclear'):
        #     continue
        magnet_info = list(map(int, magnet_info))
        add_magnet_status(df, magnet_info)

        # Calculate force on beads
        calibration = df_general_info['calibration (mV)'].values[idx]
        calculate_force(df, calibration)
        # add information about MTs 
        df['MT_STATUS']= df_general_info['MTs'].values[idx]
        df.attrs['COMMENTS'] = str(df_general_info["comments"].values[idx])

        # save the extended df to a file (in analysis folder)
        
        df.to_hdf(filepath_extended_df, key='df', mode='w', format='table')
        # Save DataFrame to HDF5 file with comments
        with pd.HDFStore(filepath_extended_df, mode='w') as store:
            store.put('df', df)
            store.get_storer('df').attrs.metadata = df.attrs['COMMENTS']

        # except:
        #     print(f'{filename} did not go through :(')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    main(args.config)