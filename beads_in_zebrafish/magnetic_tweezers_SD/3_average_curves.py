
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
    mt_codes = config['mt_codes']
    conditions = config['conditions']
    color_palette = sns.color_palette(config['color_palette'])

    creep_only = False
    if config['fit_type'] == 'creep jeff':
        creep_only = True  # If the fit type is 'creep jeff', we only plot creep curves
        
    creep_only_label = '_only_creep' if creep_only else ''

    displacement_column = 'CORRECTED DISPLACEMENT [um]' if subtract_backgroung else 'DISPLACEMENT [um]'  # Use the original displacement, not corrected
    subtr_bck_label = 'subtr_bck' if subtract_backgroung else 'no_subtr_bck'

    df_all_tracks = pd.read_csv(dir_plots + f'/results/all_full_tracks_{subtr_bck_label}.csv')

    # load properties data
    df_results_all = pd.read_csv(dir_plots + f'/results/results_material_properties_{subtr_bck_label}.csv')
    df_results_all.sort_values(by=['MT_STATUS', 'TRACK_ID'], ascending=False, inplace=True)
    df_results_filtered = filter_based_on_recovery(df_results_all, mt_codes)
    print(f'After filtering based on relative recovery we are left with {100*len(df_results_filtered)/len(df_results_all)}% of the data for calculated parameters.')

    df_all_tracks['USABLE'] = 0
    for filename in df_all_tracks['file'].unique():
        df_file = df_all_tracks[df_all_tracks['file']==filename]
        for track_id in df_file['TRACK_ID'].unique():
            df_track = df_file[df_file['TRACK_ID']==track_id]
            for pulse_n in df_track['PULSE_NUMBER'].unique():
                if len(df_results_filtered.loc[(df_results_filtered['file']==filename)&(df_results_filtered['TRACK_ID']==track_id)&(df_results_filtered['PULSE_NUMBER']==pulse_n), 'R_SQUARED_smooth'].values)>0:
                    df_all_tracks.loc[(df_all_tracks['file']==filename)&(df_all_tracks['TRACK_ID']==track_id)&(df_all_tracks['PULSE_NUMBER']==pulse_n), 'USABLE']=1
    
    print(len(df_all_tracks[df_all_tracks['USABLE']==1])/len(df_all_tracks))

    df_all_tracks = df_all_tracks.sort_values(by=['TRACK_ID', 'NORMALIZED_TIME'], ascending=True)

    df_results_from_averaged_tracks = pd.DataFrame(columns=['EMBRYO', 'MT_STATUS', 'AVG_FORCE', 'MAGNET_STATUS', 'COUNT'])

    # Tracks per embryo together
    for force_range in force_ranges:

        df_force_range = df_all_tracks[(df_all_tracks['AVG_FORCE']>=force_range[0])&(df_all_tracks['AVG_FORCE']<force_range[1])&(df_all_tracks['USABLE']==1)]

        # Average tracks per embryo (filtered data)
        df_grouped_tracks_mean = df_force_range.groupby(['EMBRYO', 'MT_STATUS', 'NORMALIZED_TIME']).mean(numeric_only=True).reset_index()
        df_grouped_tracks_mean['COUNT'] = df_force_range.groupby(by=['EMBRYO', 'MT_STATUS', 'NORMALIZED_TIME']).count()['LABEL'].values
        df_grouped_tracks_mean.to_csv(f'{dir_plots}/results/averaged_curves_per_embryo/averaged_curves_{subtr_bck_label}{creep_only_label}.csv', index=False)
        
        
        for embryo in df_force_range['EMBRYO'].unique():
            # group data by MT status
            df_embryo_filtered = df_force_range[df_force_range['EMBRYO']==embryo]

            df_grouped_mean = df_embryo_filtered.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).mean(numeric_only=True).reset_index()
            df_grouped_std = df_embryo_filtered.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).sem(numeric_only=True).reset_index()
            df_grouped_mean['COUNT'] =df_embryo_filtered.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).count()['LABEL'].values

            plt.figure(figsize=(8, 5), dpi=200)
            for (mt_status, label, i) in zip(mt_codes, conditions, [0, 1]):
                count = len(df_embryo_filtered[(df_embryo_filtered['EMBRYO']==embryo)&(df_embryo_filtered['MT_STATUS']==mt_status)].groupby(['TRACK_ID', 'PULSE_NUMBER']))
                time_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, 'NORMALIZED_TIME'].values

                if len(time_grouped) == 0:
                    continue
                displacement_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, displacement_column].values

                displacement_grouped_std = df_grouped_std.loc[df_grouped_std['MT_STATUS']==mt_status, displacement_column].values
                average_force_grouped = np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values)
                plt.plot(time_grouped, displacement_grouped, '.', color=color_palette[i], label= label)

                plt.fill_between(
                time_grouped,
                displacement_grouped - displacement_grouped_std, 
                displacement_grouped + displacement_grouped_std,
                color=color_palette[i], alpha=0.2)

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
                new_line = {'EMBRYO': embryo,
                            'MT_STATUS': mt_status,
                            'COUNT': count,
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
                for file in df_embryo_filtered['file'].unique():
                    for track in df_embryo_filtered[df_embryo_filtered['file']==file]['TRACK_ID'].unique():
                        for pulse in df_embryo_filtered[(df_embryo_filtered['file']==file)&(df_embryo_filtered['TRACK_ID']==track)]['PULSE_NUMBER'].unique():
                            df = df_embryo_filtered[(df_embryo_filtered['EMBRYO']==embryo)&(df_embryo_filtered['file']==file)&(df_embryo_filtered['TRACK_ID']==track)&(df_embryo_filtered['PULSE_NUMBER']==pulse)]
                            df = df.sort_values(by='NORMALIZED_TIME', ascending=True)
                            if len(df['MT_STATUS']) == 0:
                                continue
                            if df['MT_STATUS'].unique() == mt_codes[0]:
                                plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=color_palette[0], alpha=0.05)
                            elif df['MT_STATUS'].unique() == mt_codes[1]:
                                plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=color_palette[1], alpha=0.05)

            plt.xlabel('Time (s)')
            plt.ylabel('Displacement ($\\mathrm{\\mu}$m)')
            plt.title(f'Force range {force_range} pN')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{dir_plots}/results/averaged_curves_per_embryo/result_displacement_curve_averaged_{embryo}_{subtr_bck_label}{creep_only_label}.svg', format='svg')

    df_results_from_averaged_tracks.to_csv(f'{dir_plots}/results/results_parameters_curve_averaged_force_range{force_range[0]}-{force_range[1]}pN_{subtr_bck_label}{creep_only_label}.csv', index=False)


    # Alll averaged tracks
    for force_range in force_ranges:

        df_force_range = df_all_tracks[(df_all_tracks['AVG_FORCE']>=force_range[0])&(df_all_tracks['AVG_FORCE']<force_range[1])&(df_all_tracks['USABLE']==1)]

        df_data_filtered = df_force_range
        df_grouped_mean = df_data_filtered.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).mean(numeric_only=True).reset_index()
        df_grouped_std = df_data_filtered.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).sem(numeric_only=True).reset_index()
        df_grouped_mean['COUNT'] =df_data_filtered.groupby(by=['MT_STATUS', 'NORMALIZED_TIME']).count()['LABEL'].values

        plt.figure(figsize=(6, 5), dpi=200)
        for (mt_status, label, i) in zip(mt_codes, conditions, [0, 1]):
            count = len(df_data_filtered[(df_data_filtered['MT_STATUS']==mt_status)].groupby(['TRACK_ID', 'PULSE_NUMBER']))
            time_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, 'NORMALIZED_TIME'].values

            if len(time_grouped) == 0:
                continue
            displacement_grouped = df_grouped_mean.loc[df_grouped_mean['MT_STATUS']==mt_status, displacement_column].values

            displacement_grouped_std = df_grouped_std.loc[df_grouped_std['MT_STATUS']==mt_status, displacement_column].values
            average_force_grouped = np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values)
            plt.plot(time_grouped, displacement_grouped, '.-', color=color_palette[i], label= label)

            plt.fill_between(
            time_grouped,
            displacement_grouped - displacement_grouped_std, 
            displacement_grouped + displacement_grouped_std,
            color=color_palette[i], alpha=0.2)

            ##### fit ######
            t_1 = time_grouped[np.argmax(displacement_grouped)]
            x_fit = np.linspace(0, np.max(time_grouped), len(time_grouped))

            # weights for a better fit
            sigma = np.ones_like(displacement_grouped)
            sigma[:4] = 0.3
            window = 8
            sigma[7:12] = 0.3*np.ones(5)
            if creep_only:
                sigma[np.argmax(displacement_grouped):] = np.ones(len(time_grouped)-np.argmax(displacement_grouped))*np.inf
            
            yfit, popt, pcov = get_fit_jeff_full(x_fit, time_grouped, displacement_grouped,
                            np.average(df_grouped_mean.loc[(df_grouped_mean['MT_STATUS']==mt_status)&(df_grouped_mean['MAGNET_STATUS']==1), 'FORCE [pN]'].values), t_1, sigma=sigma)
            
            plt.plot(x_fit, yfit, 'k-', linewidth=1, alpha=0.8, zorder=10)

            plt.text(1.05, 0.1 + 0.7*i, f'Interphase \n $k$ = {popt[0]:.2f} $\\pm$ {np.sqrt(pcov[0][0]):.2f} pN/$\\mathrm{{\\mu m}}$\n$\\gamma_1$ = {popt[1]:.2f} $\\pm$ {np.sqrt(pcov[1][1]):.2f}  pN s/$\\mathrm{{\\mu m}}$\n$\\gamma_2$ = {popt[2]:.2f} $\\pm$ {np.sqrt(pcov[2][2]):.2f}  pN s/$\\mathrm{{\\mu m}}$', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))


            # plot all tracks 
            for file in df_data_filtered['file'].unique():
                for track in df_data_filtered[df_data_filtered['file']==file]['TRACK_ID'].unique():
                    for pulse in df_data_filtered[(df_data_filtered['file']==file)&(df_data_filtered['TRACK_ID']==track)]['PULSE_NUMBER'].unique():
                        df = df_data_filtered[(df_data_filtered['file']==file)&(df_data_filtered['TRACK_ID']==track)&(df_data_filtered['PULSE_NUMBER']==pulse)]
                        df = df.sort_values(by='NORMALIZED_TIME', ascending=True)
                        if len(df['MT_STATUS']) == 0:
                            continue
                        if df['MT_STATUS'].unique() == mt_codes[0]:
                            plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=color_palette[0], alpha=0.01)
                        elif df['MT_STATUS'].unique() == mt_codes[1]:
                            plt.plot(df['NORMALIZED_TIME'], df[displacement_column], '-', color=color_palette[1], alpha=0.01)

            plt.xlabel('Time (s)')
            plt.ylabel('Displacement ($\\mathrm{\\mu}$m)')
            plt.title(f'Force range {force_range} pN')
            plt.legend()
            plt.ylim(0, 12)
            plt.xlim(left=0)
            plt.tight_layout()
            plt.savefig(f'{dir_plots}/results/result_displacement_curve_averaged_all_{subtr_bck_label}{creep_only_label}.svg', format='svg')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process magnetic tweezers data and generate extended measurements.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Run the main function with the provided config file
    args = parser.parse_args()
    main(args.config)
    print('All done! :)')
