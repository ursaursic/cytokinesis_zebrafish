'''
Ursa Ursic, last updated: 22.1.2024

This code calculates displacement curves from the trackmate csv files (for high time precision measurements).
First, it calculates distances of bead to the tip end, then calculates the force on each bead. If necesary, it can substract the background flow of the beads by substracting bead mnotion in the last two thirds of the period when the magnet is off.  

use pol_stats environment: conda activate pol_stats
'''

import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import colorcet as cc
from tqdm import tqdm

import bokeh.io
import bokeh.plotting
import bokeh.models
import iqplot

from magnetic_tweezers_time_prec.utils_tp import *

intermediate_plots = True
# Connect to cytokinesis-zebrafish-collab server from your computer
general_info_csv_filepath = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers_time_prec/2_analysis/time_prec_phases_tweezers_info.csv'

# Define the result directory
plots_dir = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers_time_prec/3_plots/'
analysis_dir = '/Volumes/cytokinesis-zebrafish-collab/magnetic_tweezers_time_prec/2_analysis/'


def main():
    # General info dataframe
    df_general_info = pd.read_csv(general_info_csv_filepath, delimiter=';')

    # # Results data frame
    # df_all_results = pd.DataFrame(columns=['FILENAME', 'TRACK_IDX', 'PULSE_START_FRAME', 'MAGNET_STATUS', 'PHASE', 'VISCOSITY'])

    for idx in tqdm(range(len(df_general_info))):
        # print(f'working on file {idx}')

        # Define the path of your .csv file with tracks
        filepath=df_general_info['trackmate_file'].values[idx]
        tip_file = df_general_info['before_file'].values[idx]

        filename = os.path.basename(filepath).split('.')[0]  # This gets filename without extention
        df = pd.read_csv(filepath, skiprows=[1, 2, 3]) # skiprows to get rid of the extensive header
        df = df.sort_values(by='FRAME')
        
        # Extract data about the measurement
        first_pulse, t_on, t_off = df_general_info.loc[df_general_info['trackmate_file']==filepath, ['first_pulse (frame)', 't_on (frame)',	't_off (frame)']].values[0]
        dt = 5 #s (this is true for the 30sON, 90sOFF measurements)

        # number of pulses in total
        N_pulses = np.max(df['FRAME'])//(t_on+t_off)
        # Add info about the magnet into the df
        magnet_pulses = np.array([[first_pulse + j*(t_on+t_off) + i - 1 for i in range(0, t_on)] for j in range(N_pulses+1)])
        df['MAGNET_STATUS'] = [1 if df['FRAME'].values[i] in magnet_pulses else 0 for i in range(len(df))]

        # Detect tip
        tip_thresh = int(df_general_info['tip_threshold'].values[idx])
        tip_outline, tip_end = find_tip(tip_file, tip_thresh, save_to_file=f'{plots_dir}detecting_tips/tip_{filename}.html')
        if tip_end==False:
            continue

        add_distance_from_tip(df, tip_end) # calculate distance from the end point of the tip (not the nearest point)

        calibration = df_general_info['calibration (mV)'].values[idx]
        add_force(df, calibration) # add force from calibration params to the dataframe

        calculate_displacement(df, first_pulse, t_on, t_off, substract_background=False) # calculate displacement for every pulse

        if intermediate_plots:
            plot_trajectories(filename, df, first_pulse, t_on, t_off, save_to_filepath=f'{plots_dir}intermediate_plots/trajectories_{filename}.html')
            plot_displacement(filename, df, dt, save_to_filepath=f'{plots_dir}intermediate_plots/displacement_{filename}.html')
            # plot trajectories one over the other
            # fit the curves while ON
            # fit the curves while OFF

            # plot_displacement_force_ratio(filename, df, dt, first_pulse, t_on, t_off, save_to_filepath=f'{plots_dir}intermediate_plots/viscosity_fit_{filename}.html')

        # calculate_viscosity(df, magnet_pulses) # calculates effective viscosity when magnet is on
        # add_phase(df, phases) # add information about phases

        # # all results so far
        # df_all_results = get_results(filename, df, magnet_pulses, df_all_results)

    # df_all_results.to_csv(f'{analysis_dir}results_viscosity.csv')


if __name__=="__main__":
    main()
    print("Done! :)")