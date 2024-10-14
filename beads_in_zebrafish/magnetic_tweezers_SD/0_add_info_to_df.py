"""
Author: Ursa Ursic
Last updated: 28.7.2024

This script processes magnetic tweezers data and generates CSV files with additional information:
- Distance from tip
- Force calculation
- Magnet pulses information
- Microtubule (MT) status

It also saves images of the tip with outlines. This script only needs to be run once. Further analysis is done in a different file.
"""

import pandas as pd
import os
from tqdm import tqdm
import argparse
import yaml
from utils import *  # Custom utility functions, assumed to include methods like find_tip, calculate_force, etc.


def main(config_path):
    """
    Main function to process data and generate extended measurements.

    Parameters:
    config_path (str): Path to the YAML configuration file.
    """

    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract paths and configuration settings from the loaded YAML
    filepath_measurements_info = config['filepath_measurements_info']
    dir_plots = os.path.join(config['dir_parent'], '3_plots')  # Folder for saving plots
    dir_analysis = os.path.join(config['dir_parent'], '2_analysis')  # Folder for saving analysis results
    recalculate = config['recalculate']  # Whether to recalculate measurements if they already exist

    # Directory where extended measurement data will be saved
    dir_measurements_extended = os.path.join(dir_analysis, 'measurements_extended_info')

    # Load general measurement info CSV
    df_general_info = pd.read_csv(filepath_measurements_info, delimiter=';', encoding='utf-8')

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_measurements_extended):
        os.mkdir(dir_measurements_extended)
    
    # Process each measurement in the dataset
    for idx in tqdm(range(len(df_general_info))):
        # Load data for each measurement
        filepath = df_general_info['trackmate_file'].values[idx]
        filename = os.path.basename(filepath).split('.')[0]  # Extract file name without extension
        
        # Define the path where the extended data will be saved
        filepath_extended_df = os.path.join(dir_measurements_extended, f'{filename}_extended.h5')

        # Skip if the extended file already exists and recalculation is not needed
        if os.path.exists(filepath_extended_df) and not recalculate:
            continue

        # Load raw data from the measurement file
        df = pd.read_csv(filepath, skiprows=[1, 2, 3], encoding="utf-8")

        # Check if tip information is available and process accordingly
        if 'before_file' in df_general_info.columns:
            # Load the tip image, save the outline, and calculate distance from the tip
            filepath_tip = df_general_info['before_file'].values[idx]
            threshold_tip = df_general_info['tip_threshold'].values[idx]
            dir_tip_imgs = os.path.join(dir_plots, 'tip_images')
            if not os.path.exists(dir_tip_imgs):
                os.mkdir(dir_tip_imgs)
            filepath_tip_outline = os.path.join(dir_tip_imgs, f'{filename}.png')

            # Find the tip position and save the image with tip outline
            tip = find_tip(filepath_tip, threshold_tip, save_img_to_path=filepath_tip_outline, endpoint=True)

        elif 'tip_x' in df_general_info.columns:
            # If tip position is already available, use it directly
            tip = list(df_general_info[['tip_x', 'tip_y']].values[idx])

        # Calculate distance from tip if tip information is available
        if tip:
            calculate_distance_from_tip(df, tip) 
        else:
            # Skip if no tip information is found
            continue

        # Add information about magnet pulse status to the dataframe
        magnet_info = df_general_info[['first_pulse (frame)', 't_on (frame)', 't_off (frame)']].values[idx]
        magnet_info = list(map(int, magnet_info))  # Convert magnet info to integer
        add_magnet_status(df, magnet_info)

        # Calculate force on the beads using calibration values
        calibration = df_general_info['calibration (mV)'].values[idx]
        calculate_force(df, calibration)

        # Add microtubule status information to the dataframe
        df['MT_STATUS'] = df_general_info['MTs'].values[idx]

        # Calculate displacement, with and without drift correction
        df = add_calculated_displacement(df)

        # Add comments from the general info to the dataframe's metadata
        df.attrs['COMMENTS'] = str(df_general_info["comments"].values[idx])

        # Save the extended dataframe as an HDF5 file, including metadata
        with pd.HDFStore(filepath_extended_df, mode='w') as store:
            store.put('df', df)
            store.get_storer('df').attrs.metadata = df.attrs['COMMENTS']


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process magnetic tweezers data and generate extended measurements.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Run the main function with the provided config file
    args = parser.parse_args()
    main(args.config)