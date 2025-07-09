'''
use the environment: cytokinesis_zeb

Analyze trajectories, fit Jeffrey's model and Kelvin Voigt model, analyze model independent parameters for trajectories. Save data to a csv file with all the auxiliary parameters, lik R squared and so on.
'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
import yaml
import matplotlib.pyplot as plt

from utils import *


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    filepath_measurements_info = config['filepath_measurements_info']
    dir_plots = os.path.join(config['dir_parent'], '3_plots')  # Folder for saving plots
    dir_analysis = os.path.join(config['dir_parent'], '2_analysis')  # Folder for saving analysis results
    dt = config['dt']
    plot = config['plot']
    subtract_background = config['subtract_background']  # Whether to subtract background from displacement
    subtr_bck_label = 'subtr_bck' if subtract_background else 'no_subtr_bck'

    displacement_column = 'CORRECTED DISPLACEMENT [um]' if subtract_background else 'DISPLACEMENT [um]'

    df_general_info = pd.read_csv(filepath_measurements_info, delimiter=';', encoding='utf-8')
    df_general_info = df_general_info.sort_values(by='trackmate_file')

    dir_measurements_extended = dir_analysis+'/measurements_extended_info/'

    df_results = pd.DataFrame()
    df_all_tracks = pd.DataFrame()

    for filename in tqdm(os.listdir(dir_measurements_extended)):
        df_all = pd.read_hdf(dir_measurements_extended+filename, key='df')
        df_all = df_all.sort_values(by='POSITION_T')

        t_on, t_off = 10, 30

        # filter tracks
        df = filter_data_model_independent(df_all, t_on, t_off, displacement_column)
        df['file'] = filename
        df['EXPERIMENT'] = filename[:18].replace('_', '')
        df['EMBRYO'] = filename[:15].replace('_', '')
        

        for track_id in df['TRACK_ID'].unique():
            track = df[df['TRACK_ID']==track_id]
            for pulse_n in track['PULSE_NUMBER'].unique():
                pulse = track[track['PULSE_NUMBER']==pulse_n]
                
                time_on = pulse.loc[pulse['MAGNET_STATUS']==1, 'FRAME'].values * dt
                time_off = pulse.loc[pulse['MAGNET_STATUS']==0, 'FRAME'].values * dt
                
                # normalize time to start at 0
                time_off -= time_on[0]
                time_on -= time_on[0]
                time_data = np.concatenate((time_on, time_off))

                displacement_magnet_on = pulse.loc[pulse['MAGNET_STATUS']==1, displacement_column].values
                displacement_magnet_off = pulse.loc[pulse['MAGNET_STATUS']==0, displacement_column].values
                displacement_full = np.concatenate((displacement_magnet_on, displacement_magnet_off))

                avg_force = np.average(pulse.loc[pulse['MAGNET_STATUS']==1, 'FORCE [pN]'].values)
                t_1 = time_on[-1]

                df.loc[(df['TRACK_ID']==track_id)&(df['PULSE_NUMBER']==pulse_n), 'NORMALIZED_TIME'] = time_data
                df.loc[(df['TRACK_ID']==track_id)&(df['PULSE_NUMBER']==pulse_n), 'AVG_FORCE'] = avg_force
                
                r2, smooth_full = get_r2_from_smooth_curves(time_on, time_off, displacement_magnet_on, displacement_magnet_off)
                
                if plot:
                    fig = plt.figure(figsize=(10, 7))
                    plt.plot(time_data, displacement_full, 'k-', alpha = 0.5)
                    plt.plot(time_on, displacement_magnet_on, 'o', color ='green', alpha=0.5, label = 'magnet ON')
                    plt.plot(time_off, displacement_magnet_off, 'o', color='gray', alpha=0.5, label = 'magnet OFF')
                    plt.plot(time_data, smooth_full, 'r-', alpha=0.5, label='smooth curve')

                if config['fit_type'] == 'full jeff':
                    # weights for a better fit
                    sigma = np.ones_like(displacement_full)
                    sigma[:4] = 0.3
                    window = 8
                    sigma[len(time_on) - int(window//4):len(time_on) + int(3*window//4)] = 0.3*np.ones(int(window//4) + int(3*window//4))
                    sigma[len(time_on)-1:len(time_on)+2] = 0.1*np.ones(3)

                    params = calculate_Jeff_fit_params(time_data, displacement_full, avg_force, t_1, dt, sigma, plot=plot)

                sigma = np.ones_like(displacement_full)

                k, eta_1, eta_2, k_err, eta_1_err, eta_2_err, r2_fit = params

                if plot:
                    plt.xlabel('Time (s)')
                    plt.ylabel('Displacement (um)')
                    plt.title(f'{filename.split("_")[0]} {filename.split("_")[1][0:9]}, track_ID: {track_id}, MT: {pulse["MT_STATUS"].values[0]}, force: {int(avg_force)} pN')
                    plt.legend()
                    plt.xlim(left=0)
                    plt.ylim(bottom=0)
  
                    if not os.path.exists(dir_plots + f'/all_fits/'):
                        os.makedirs(dir_plots + f'/all_fits/')
                    
                    try:
                        plt.savefig(dir_plots + f'/all_fits/Jeff_fit_{filename.split("_")[0]}_{filename.split("_")[1][0:9]}_track_ID_{track_id}_pulse_n_{pulse_n}_{subtr_bck_label}.png', dpi=300)
                        plt.close()
                    except:
                        print(f'error at {filename.split("_")[0]}_{filename.split("_")[1][0:9]}_track_ID_{track_id}_pulse_n_{pulse_n}')
                
                # include model independent analysis
                rising_dif, relaxing_dif, rising_dif_norm, rising_dif_norm_inverse = calculate_model_independedt_params(pulse, avg_force, subtract_background=subtract_background)

                new_line = {'file': filename,
                            'EXPERIMENT': [filename[:18].replace('_', '')], 
                            'EMBRYO': [filename[:15].replace('_', '')],
                            'TRACK_ID': track_id, 
                            'PULSE_NUMBER': pulse_n, 
                            'MT_STATUS': pulse['MT_STATUS'].unique(), 
                            'AVG_FORCE': avg_force,
                            'k': k, 
                            'eta_1': eta_1, 
                            'eta_2': eta_2, 
                            'k_err': k_err, 
                            'eta_1_err': eta_1_err, 
                            'eta_2_err': eta_2_err,
                            't_1': t_1,
                            'R_SQUARED_smooth': r2, 
                            'R_SQUARED_fit': r2_fit,
                            'rising_dif': [rising_dif], 
                            'relaxing_dif': [relaxing_dif], 
                            'rising_dif_norm': [rising_dif_norm], 
                            'rising_dif_norm_inverse': [rising_dif_norm_inverse],
                            'relative_dif': [-relaxing_dif/rising_dif],
                            'tau_r': eta_1/(2*k), 
                            'a': 1 - 1 / ((eta_2 / (k * t_1)) * (1- np.exp(- k * t_1 / eta_1)) +1),
                            'elastic_viscous_ratio': (avg_force / k * (1 - np.exp(- k  * t_1  / eta_1))) / (avg_force * t_1 / eta_2)
                            }

                df_results = pd.concat([df_results, pd.DataFrame(new_line, index=[0])], ignore_index=True)
        
        df_all_tracks = pd.concat([df_all_tracks, df])
  
    df_results.to_csv(dir_plots + f'/results/results_material_properties_{subtr_bck_label}.csv')
    df_all_tracks.to_csv(dir_plots + f'/results/all_full_tracks_{subtr_bck_label}.csv')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process magnetic tweezers data and generate extended measurements.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Run the main function with the provided config file
    args = parser.parse_args()
    main(args.config)