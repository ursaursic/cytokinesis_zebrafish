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
import seaborn as sns
from scipy.signal import find_peaks
import sys

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

    count_failed = 0
    for filename in tqdm(os.listdir(dir_measurements_extended)):
        df_all = pd.read_hdf(dir_measurements_extended+filename, key='df')
        df_all = df.sort_values(by='POSITION_T')

        # filter tracks
        df = filter_data_model_independent(df_all)

        for track_id in df['TRACK_ID'].unique():
            track = df[df['TRACK_ID']==track_id]
            for pulse_n in track['PULSE_NUMBER'].unique():

                pulse = track[track['PULSE_NUMBER']==pulse_n]
                time_on = pulse.loc[pulse['MAGNET_STATUS']==1, 'FRAME'].values * dt
                time_off = pulse.loc[pulse['MAGNET_STATUS']==0, 'FRAME'].values * dt

                if len(time_on) < 4 or len(time_off) < 10:
                    count_failed += 1
                    continue
                
                # normalize time to start at 0
                time_off -= time_on[0]
                time_on -= time_on[0]
                time_data = np.concatenate((time_on, time_off))

                displacement_magnet_on = pulse.loc[pulse['MAGNET_STATUS']==1, displacement_column].values
                displacement_magnet_off = pulse.loc[pulse['MAGNET_STATUS']==0, displacement_column].values

                # check if too many data points go below zero during relaxation
                relax_too_much = False
                if sum(displacement_magnet_off < 0)/len(displacement_magnet_off) > 0.5:
                    relax_too_much = True
                displacement_full = np.concatenate((displacement_magnet_on, displacement_magnet_off))
                if np.all(np.isnan(displacement_full)):
                    count_failed += 1
                    continue

                avg_force = np.average(pulse.loc[pulse['MAGNET_STATUS']==1, 'FORCE [pN]'].values)

                t_1 = time_on[-1]
                
                # Soemtimes the magnet was still on after the last frame 
                if displacement_full[len(time_on)] > displacement_full[(len(time_on) - 1)]:
                    t_1 = time_off[0]
                
                if plot:
                    fig = plt.figure(figsize=(10, 7))
                    plt.plot(time_data, displacement_full, 'k-', alpha = 0.5)
                    plt.plot(time_on, displacement_magnet_on, 'o', color ='green', alpha=0.5, label = 'magnet ON')
                    plt.plot(time_off, displacement_magnet_off, 'o', color='gray', alpha=0.5, label = 'magnet OFF')

                if config['fit_type'] == 'full jeff':
                    # weights for a better fit
                    sigma = np.ones_like(displacement_full)
                    sigma[:4] = 0.3
                    window = 8
                    sigma[len(time_on) - int(window//4):len(time_on) + int(3*window//4)] = 0.3*np.ones(int(window//4) + int(3*window//4))
                    sigma[len(time_on)-1:len(time_on)+2] = 0.1*np.ones(3)

                    params = calculate_Jeff_fit_params(time_data, displacement_full, avg_force, t_1, dt, sigma, plot=plot)

                elif config['fit_type'] == 'full KV':
                    sigma = np.ones_like(displacement_full)
                    params_KV = calculate_KV_fit_params(time_data, displacement_full, avg_force, t_1, dt, sigma, plot=plot)
                
                elif config['fit_type'] == 'creep jeff':
                    sigma = np.ones_like(displacement_magnet_on)
                    params = calculate_Jeff_fit_params(time_on, displacement_magnet_on, avg_force, t_1, dt, sigma, plot=plot)

                if config['fit_type'] != 'full KV':
                    sigma = np.ones_like(displacement_full)
                    params_KV = calculate_KV_fit_params(time_data, displacement_full, avg_force, t_1, dt, sigma, plot=False)

                k, eta_1, eta_2, k_err, eta_1_err, eta_2_err, R_sq = params
                k_KV, eta_KV, k_KV_err, eta_KV_err, R_sq_KV = params_KV

                fit_divisible = check_differentiable(time_data, k, eta_1, eta_2, avg_force, t_1, dt)

                if plot:
                    plt.xlabel('Time (s)')
                    plt.ylabel('Displacement (um)')
                    plt.title(f'{filename.split("_")[0]} {filename.split("_")[1][0:9]}, track_ID: {track_id}, MT: {pulse["MT_STATUS"].values[0]}, force: {int(avg_force)} pN, relax_too_much: {relax_too_much}')
                    plt.legend()
                    plt.xlim(left=0)
                    plt.ylim(bottom=0)
                    # plt.yscale('log')
                    # plt.xscale('log')
                    conditions = (R_sq >= 0.5) & (not relax_too_much) & (fit_divisible) & (k_err/k < 1) & (eta_1_err/eta_1 < 1) & (eta_2_err/eta_2 < 1)  
                    if not os.path.exists(dir_plots + f'/all_fits/'):
                        os.makedirs(dir_plots + f'/all_fits/')
                    if conditions:
                        plt.savefig(dir_plots + f'/all_fits/Jeff_fit_{filename.split("_")[0]}_{filename.split("_")[1][0:9]}_track_ID_{track_id}_pulse_n_{pulse_n}_{subtr_bck_label}.png', dpi=300)
                    else: 
                        plt.savefig(dir_plots + f'/all_fits/discarded/Jeff_fit_{filename.split("_")[0]}_{filename.split("_")[1][0:9]}_track_ID_{track_id}_pulse_n_{pulse_n}_{subtr_bck_label}.png', dpi=300)
                    plt.close()

                
                
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
                            'k_KV': k_KV,
                            'eta_KV': eta_KV,
                            'k_KV_err': k_KV_err,
                            'eta_KV_err': eta_KV_err,
                            't_1': t_1,
                            'R_SQUARED': R_sq, 
                            'R_SQUARED_KV': R_sq_KV,
                            'fit_divisible': fit_divisible,
                            'relax_too_much': relax_too_much,
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
  
    df_results.to_csv(dir_plots + f'/results/results_material_properties_{subtr_bck_label}.csv')

    print('Number of failed tracks:', count_failed)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process magnetic tweezers data and generate extended measurements.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Run the main function with the provided config file
    args = parser.parse_args()
    main(args.config)
