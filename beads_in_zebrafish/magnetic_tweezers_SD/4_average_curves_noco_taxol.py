
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm  # For displaying progress bars during iteration
import yaml  # For reading YAML configuration files
import argparse  # For parsing command-line arguments

from utils import *


def main(config_path):

    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dir_plots = os.path.join(config['dir_parent'], '3_plots')  # Folder for saving plots
    dir_analysis = os.path.join(config['dir_parent'], '2_analysis')  # Folder for saving analysis results 
    dt = config['dt']  # Time step in seconds\
    force_ranges = config['force_ranges']  # Force ranges for averaging curves
    subtract_backgroung = config['subtract_background']  # Whether to subtract background from displacement

    creep_only = False
    if config['fit_type'] == 'creep jeff':
        creep_only = True  # If the fit type is 'creep jeff', we only plot creep curves
        
    creep_only_label = '_only_creep' if creep_only else ''

    displacement_column = 'CORRECTED DISPLACEMENT [um]' if subtract_backgroung else 'DISPLACEMENT [um]'  # Use the original displacement, not corrected
    subtr_bck_label = 'subtr_bck' if subtract_backgroung else 'no_subtr_bck'

    folder = dir_analysis + '/measurements_extended_info/'

    df_all_tracks = pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith('.h5'):
            # Read the data using pandas
            df = pd.read_hdf(folder+file)
            df['file'] = file
            df['EXPERIMENT'] = file[:18].replace('_', '')
            df['EMBRYO'] = file[:15].replace('_', '')
            df_all_tracks = pd.concat([df_all_tracks, df])

    df_all_tracks.dropna(subset=[displacement_column], inplace=True)

    # load data
    df_results_all = pd.read_csv(dir_plots + f'/results/results_material_properties_{subtr_bck_label}.csv')
    df_results_all.sort_values(by='MT_STATUS', ascending=False, inplace=True)

    r_sq_min = 0.5
    # filter out the unreasonable tracks
    df_results_filtered = df_results_all[
        (df_results_all['R_SQUARED'] >= r_sq_min) &
        (df_results_all['relax_too_much'] == False) &
        (df_results_all['fit_divisible'] == True) &
        (df_results_all['k_err']/df_results_all['k'] < 1) &
        (df_results_all['eta_1_err']/df_results_all['eta_1'] < 1) &
        (df_results_all['eta_2_err']/df_results_all['eta_2'] < 1)  
    ]

    df_all_tracks['USABLE'] = np.zeros(len(df_all_tracks))
    df_all_tracks.sort_values(by='FRAME')
    for measurement in df_results_filtered[['file', 'TRACK_ID', 'PULSE_NUMBER']].values:
        file, track, pulse = measurement
        
        df = df_all_tracks.loc[(df_all_tracks['file']==file)&(df_all_tracks['TRACK_ID']==track)&(df_all_tracks['PULSE_NUMBER']==pulse)]

        if len(df['POSITION_T'])>50 or len(df[df['MAGNET_STATUS']==1])<10:
            plt.plot(df['POSITION_T']-np.min(df['POSITION_T']), df[displacement_column], 'o', color=sns.color_palette()[0], alpha=0.5, markersize = 4)
            print('Something went wrong.')
            continue

        df_all_tracks.loc[(df_all_tracks['file']==file)&(df_all_tracks['TRACK_ID']==track)&(df_all_tracks['PULSE_NUMBER']==pulse), 'NORMALIZED_TIME'] = (df['FRAME']-np.min(df['FRAME'])) * dt
        
        df_all_tracks.loc[(df_all_tracks['file']==file)&(df_all_tracks['TRACK_ID']==track)&(df_all_tracks['PULSE_NUMBER']==pulse), 'AVG_FORCE'] = df_results_filtered.loc[(df_results_filtered['file']==file)&(df_results_filtered['TRACK_ID']==track)&(df_results_filtered['PULSE_NUMBER']==pulse), 'AVG_FORCE'].values[0]

        df_all_tracks.loc[(df_all_tracks['file']==file)&(df_all_tracks['TRACK_ID']==track)&(df_all_tracks['PULSE_NUMBER']==pulse), 'USABLE'] = 1

    # all tracks 

    df_all_tracks = df_all_tracks.sort_values(by=['TRACK_ID', 'NORMALIZED_TIME'], ascending=True)

    for force_range in force_ranges:
        df_force_range = df_all_tracks[(df_all_tracks['AVG_FORCE']>=force_range[0])&(df_all_tracks['AVG_FORCE']<force_range[1])&(df_all_tracks['USABLE']==1)]
        # group data by MT status
        df_grouped_mean = df_force_range.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).mean(numeric_only=True).reset_index()
        df_grouped_std = df_force_range.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).sem(numeric_only=True).reset_index()

        df_force_range.dropna(subset='NORMALIZED_TIME')
        df_grouped_tracks_mean = df_force_range.groupby(['EMBRYO', 'MT_STATUS', 'NORMALIZED_TIME']).mean(numeric_only=True).reset_index()
        df_grouped_tracks_mean['COUNT'] = df_force_range.groupby(by=['EMBRYO', 'MT_STATUS', 'NORMALIZED_TIME']).count()['LABEL'].values

        df_grouped_tracks_mean.to_csv(f'{dir_plots}/results/averaged_curves_per_embryo/averaged_curves_{subtr_bck_label}{creep_only_label}.csv', index=False)

        df_grouped_mean.to_csv(f'{dir_plots}/results/results_displacement_curve_averaged_force_range{force_range[0]}-{force_range[1]}pN_{subtr_bck_label}.csv', index=False)

        plt.figure(figsize=(8, 5), dpi=200)
        for (mt_status, label, i) in zip(['taxol', 'noco'], ['Taxol', 'Nocodazole'], [0, 1]):
            time_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, 'NORMALIZED_TIME'].values

            if len(time_grouped) == 0:
                continue
            displacement_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, displacement_column].values
            displacement_grouped_std = df_grouped_std.loc[df_grouped_std['MT_STATUS']==mt_status, displacement_column].values
            average_force_grouped = np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values)
            plt.plot(time_grouped, displacement_grouped, '.', color=sns.color_palette('dark')[i], label= label)

            plt.fill_between(
            time_grouped,
            displacement_grouped - displacement_grouped_std, 
            displacement_grouped + displacement_grouped_std,
            color=sns.color_palette('dark')[i], alpha=0.2)

            ##### fit ######
            t_1 = time_grouped[np.argmax(displacement_grouped)]
            x_fit = np.linspace(0, np.max(time_grouped), len(time_grouped))

            sigma = np.ones(len(time_grouped))
            if creep_only:
                sigma[np.argmax(displacement_grouped):] = np.ones(len(time_grouped)-np.argmax(displacement_grouped))*np.inf
            
            yfit, popt, pcov = get_fit_jeff_full(x_fit, time_grouped, displacement_grouped,
                            np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values), t_1, sigma=sigma)
            
            plt.plot(x_fit, yfit, 'k-', linewidth=1, alpha=0.8, zorder=10)

            plt.text(1.05, 0.1 + 0.7*i, f'Interphase \n $k$ = {popt[0]:.2f} $\\pm$ {np.sqrt(pcov[0][0]):.2f} pN/$\\mathrm{{\\mu m}}$\n$\\gamma_1$ = {popt[1]:.2f} $\\pm$ {np.sqrt(pcov[1][1]):.2f}  pN s/$\\mathrm{{\\mu m}}$\n$\\gamma_2$ = {popt[2]:.2f} $\\pm$ {np.sqrt(pcov[2][2]):.2f}  pN s/$\\mathrm{{\\mu m}}$', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))


            ###############

            # plot all tracks 

            for file in df_force_range['file'].unique():
                for track in df_force_range[df_force_range['file']==file]['TRACK_ID'].unique():
                    for pulse in df_force_range[(df_force_range['file']==file)&(df_force_range['TRACK_ID']==track)]['PULSE_NUMBER'].unique():
                        df = df_force_range[(df_force_range['file']==file)&(df_force_range['TRACK_ID']==track)&(df_force_range['PULSE_NUMBER']==pulse)]
                        df = df.sort_values(by='NORMALIZED_TIME', ascending=True)
                        if df['MT_STATUS'].unique() == 'taxol':
                            plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=sns.color_palette('dark')[0], alpha=0.02, zorder=1)
                        elif df['MT_STATUS'].unique() == 'noco':
                            plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=sns.color_palette('dark')[1], alpha=0.02, zorder=1)

        plt.xlabel('Time (s)')
        plt.ylabel('Displacement ($\\mathrm{\\mu}$m)')
        plt.title(f'Force range {force_range} pN')
        plt.legend()
        plt.ylim(bottom=0, top=12)
        plt.xlim(left=0)
        # plt.yscale('log')
        # plt.xscale('log')
        plt.ylim(bottom=0.08)
        plt.xlim(left=0.1)
        plt.tight_layout()
        plt.savefig(f'{dir_plots}/results/result_displacement_curve_averaged_{subtr_bck_label}{creep_only_label}.svg', format='svg')


    df_all_tracks = df_all_tracks.sort_values(by=['TRACK_ID', 'NORMALIZED_TIME'], ascending=True)
    df_results_from_averaged_tracks = pd.DataFrame(columns=['EMBRYO', 'MT_STATUS', 'AVG_FORCE', 'MAGNET_STATUS'])


    for force_range in force_ranges:
        df_force_range = df_all_tracks[(df_all_tracks['AVG_FORCE']>=force_range[0])&(df_all_tracks['AVG_FORCE']<force_range[1])&(df_all_tracks['USABLE']==1)]
        
        for embryo in df_force_range['EMBRYO'].unique():
            # group data by MT status
            df_grouped_mean = df_force_range[df_force_range['EMBRYO']==embryo].groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).mean(numeric_only=True).reset_index()
            df_grouped_std = df_force_range[df_force_range['EMBRYO']==embryo].groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).sem(numeric_only=True).reset_index()

            plt.figure(figsize=(8, 5), dpi=200)
            for (mt_status, label, i) in zip(['taxol', 'noco'], ['Taxol', 'Nocodazole'], [0, 1]):
                time_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, 'NORMALIZED_TIME'].values

                if len(time_grouped) == 0:
                    continue
                displacement_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, displacement_column].values
                displacement_grouped_std = df_grouped_std.loc[df_grouped_std['MT_STATUS']==mt_status, displacement_column].values
                average_force_grouped = np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values)
                plt.plot(time_grouped, displacement_grouped, '.', color=sns.color_palette('dark')[i], label= label)

                plt.fill_between(
                time_grouped,
                displacement_grouped - displacement_grouped_std, 
                displacement_grouped + displacement_grouped_std,
                color=sns.color_palette('dark')[i], alpha=0.2)

                ##### fit ######
                t_1 = time_grouped[np.argmax(displacement_grouped)]
                x_fit = np.linspace(0, np.max(time_grouped), len(time_grouped))

                sigma = np.ones(len(time_grouped))
                if creep_only:
                    sigma[np.argmax(displacement_grouped):] = np.ones(len(time_grouped)-np.argmax(displacement_grouped))*np.inf
                
                yfit, popt, pcov = get_fit_jeff_full(x_fit, time_grouped, displacement_grouped,
                                np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values), t_1, sigma=sigma)
                
                plt.plot(x_fit, yfit, 'k-', linewidth=1, alpha=0.8, zorder=10)

                plt.text(1.05, 0.1 + 0.7*i, f'Interphase \n $k$ = {popt[0]:.2f} $\\pm$ {np.sqrt(pcov[0][0]):.2f} pN/$\\mathrm{{\\mu m}}$\n$\\gamma_1$ = {popt[1]:.2f} $\\pm$ {np.sqrt(pcov[1][1]):.2f}  pN s/$\\mathrm{{\\mu m}}$\n$\\gamma_2$ = {popt[2]:.2f} $\\pm$ {np.sqrt(pcov[2][2]):.2f}  pN s/$\\mathrm{{\\mu m}}$', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

                ###############
                
                # include model independent analysis
                # calculate R squared
                R_sq = r_squared(displacement_grouped, yfit)
                new_line = { 'EMBRYO': embryo,
                            'MT_STATUS': mt_status, 
                            'AVG_FORCE': average_force_grouped,
                            'k': popt[0], 
                            'eta_1': popt[1], 
                            'eta_2': popt[2], 
                            'k_err': np.sqrt(pcov[0][0]), 
                            'eta_1_err': np.sqrt(pcov[1][1]), 
                            'eta_2_err': np.sqrt(pcov[2][2]),
                            't_1': t_1,
                            'R_SQUARED': R_sq, 
                            'rising_dif': [np.max(displacement_grouped)], 
                            'relaxing_dif': [np.max(displacement_grouped) - displacement_grouped[-1]], 
                            'rising_dif_norm': [np.max(displacement_grouped)/average_force_grouped], 
                            'relative_dif': [(np.max(displacement_grouped) - displacement_grouped[-1])/ np.max(displacement_grouped)],
                            'tau_r': popt[1]/(2*popt[0]), 
                            'a': 1 - 1 / ((popt[2] / (popt[0] * t_1)) * (1- np.exp(- popt[0]* t_1 / popt[1])) +1),
                            'elastic_viscous_ratio': (average_force_grouped / popt[0] * (1 - np.exp(- popt[0]  * t_1  / popt[1]))) / (average_force_grouped * t_1 / popt[2])
                            }

                df_results_from_averaged_tracks = pd.concat([df_results_from_averaged_tracks, pd.DataFrame(new_line, index=[0])], ignore_index=True)

                # plot all tracks 

                for file in df_force_range['file'].unique():
                    for track in df_force_range[df_force_range['file']==file]['TRACK_ID'].unique():
                        for pulse in df_force_range[(df_force_range['file']==file)&(df_force_range['TRACK_ID']==track)]['PULSE_NUMBER'].unique():
                            df = df_force_range[(df_force_range['EMBRYO']==embryo)&(df_force_range['file']==file)&(df_force_range['TRACK_ID']==track)&(df_force_range['PULSE_NUMBER']==pulse)]
                            df = df.sort_values(by='NORMALIZED_TIME', ascending=True)
                            if len(df['MT_STATUS']) == 0:
                                continue
                            if df['MT_STATUS'].unique() == 'taxol':
                                plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=sns.color_palette('dark')[0], alpha=0.05)
                            elif df['MT_STATUS'].unique() == 'noco':
                                plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=sns.color_palette('dark')[1], alpha=0.05)

            plt.xlabel('Time (s)')
            plt.ylabel('Displacement ($\\mathrm{\\mu}$m)')
            plt.title(f'Force range {force_range} pN')
            plt.legend()
            # plt.ylim(bottom=0, top=12)
            plt.xlim(left=0)
            # plt.yscale('log')
            # plt.xscale('log')
            plt.ylim(bottom=0.08)
            plt.xlim(left=0.1)
            plt.tight_layout()
            plt.savefig(f'{dir_plots}/results/averaged_curves_per_embryo/result_displacement_curve_averaged_{embryo}_{subtr_bck_label}{creep_only_label}.svg', format='svg')

    df_results_from_averaged_tracks.to_csv(f'{dir_plots}/results/results_parameters_curve_averaged_force_range{force_range[0]}-{force_range[1]}pN_{subtr_bck_label}{creep_only_label}.csv', index=False)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process magnetic tweezers data and generate extended measurements.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Run the main function with the provided config file
    args = parser.parse_args()
    main(args.config)
