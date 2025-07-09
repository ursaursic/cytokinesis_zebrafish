import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from statsmodels.stats.weightstats import ttest_ind
from tqdm import tqdm  # For displaying progress bars during iteration
import yaml  # For reading YAML configuration files
import argparse  # For parsing command-line arguments

from utils import *


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    filepath_measurements_info = config['filepath_measurements_info']
    dir_plots = os.path.join(config['dir_parent'], '3_plots')  # Folder for saving plots
    dir_analysis = os.path.join(config['dir_parent'], '2_analysis')  # Folder for saving analysis results
    recalculate = config['recalculate']  # Whether to recalculate measurements if they already exist
    pix_size = config['pix_size'] 
    dt = config['dt']  # Time step in seconds

    params = ['k', 'eta_1', 'eta_2', 'elastic_viscous_ratio', 'rising_dif', 'rising_dif_norm', 'relative_dif', 'a', 'tau_r']

    labels = ['$\\kappa$ (pN/$\\mathrm{\\mu m}$)', '$\\gamma_1$ (pN s/$\\mathrm{\\mu m}$)', '$\\gamma_2$ (pN s/$\\mathrm{\\mu m}$)', 'Ratio between elastic and viscous deformation', 'Displacement ($\\mathrm{\\mu m}$)', 'Displacement / Force ($\\mathrm{\\mu m}$/ pN)', 'a = Relaxation / Displacement ', 'a', '$\\tau_R$']

    mapping = {'y': 'Interphase', 'n': 'M-phase', 'taxol': 'Taxol', 'noco': 'Nocodazole'}
    custom_dict = {'y': 0, 'taxol': 1, 'n': 2, 'noco': 3} 

    force_ranges = config['force_ranges']  # Force ranges for averaging curves
    subtract_backgroung = config['subtract_background']  # Whether to subtract background from displacement
    mt_codes = config['mt_codes']
    conditions = config['conditions']
    color_palette = sns.color_palette(config['color_palette'])[:2]

    creep_only = False
    if config['fit_type'] == 'creep jeff':
        creep_only = True  # If the fit type is 'creep jeff', we only plot creep curves
        
    creep_only_label = '_only_creep' if creep_only else ''

    displacement_column = 'CORRECTED DISPLACEMENT [um]' if subtract_backgroung else 'DISPLACEMENT [um]'  # Use the original displacement, not corrected
    subtr_bck_label = 'subtr_bck' if subtract_backgroung else 'no_subtr_bck'

    # load data
    df_results_all = pd.read_csv(dir_plots + f'/results/results_material_properties_{subtr_bck_label}.csv')
    df_results_all.sort_values(by='MT_STATUS', key=lambda x: x.map(custom_dict), inplace=True)
    df_results_all.head(10)


    N_tracks = len(df_results_all)
    N_tracks_i = len(df_results_all[df_results_all['MT_STATUS']==mt_codes[0]])
    N_tracks_m = len(df_results_all[df_results_all['MT_STATUS']==mt_codes[1]])
    print(f'We have in total {N_tracks} full tracks. \nOut of which:\n - {conditions[0]}: {N_tracks_i} \n - {conditions[1]}: {N_tracks_m} ')

    df_results_filtered = filter_based_on_recovery(df_results_all, mt_codes)



    df_results_filtered = df_results_filtered.sort_values(by='AVG_FORCE', ascending=True)

    # plot histogram of recovery 
    param = 'relative_dif'
    fig = plt.figure(figsize=(6, 4), dpi=200)
    df_binned_force = df_results_all[(df_results_all['AVG_FORCE']>force_ranges[0][0])&(df_results_all['AVG_FORCE']<force_ranges[0][1])]
    # Compute mean and standard deviation
    for (mt_status, color) in zip(mt_codes, [color_palette[0], color_palette[1]]):
        values = df_binned_force.loc[df_binned_force['MT_STATUS']==mt_status, param]
        ci = np.percentile(values, [5, 95])
        plt.vlines(ci[0], 0, 50, color=color)
        plt.vlines(ci[1], 0, 50, color=color)

    sns.histplot(df_binned_force, x=param, hue='MT_STATUS', palette=color_palette)
    plt.xlabel(param)
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(f'{dir_plots}/results/auxiliary_plots_statistics/result_histogram_{param}.svg', format='svg')


    df_data_stats = df_results_filtered.groupby(['EMBRYO', 'MT_STATUS']).median(numeric_only=True).reset_index()
    df_data_stats['group_size'] = df_results_filtered.groupby(['EMBRYO', 'MT_STATUS']).size().values

    rng = np.random.default_rng()
    df_params_stats = pd.DataFrame()
    p_values = dict()

    for param in params:
        df_binned_force = df_data_stats
        for mt_status in mt_codes:

            weights=df_binned_force.loc[df_binned_force['MT_STATUS']==mt_status, 'group_size'].values
            weighted_average, ci = weighted_bootstrap(df_binned_force.loc[df_binned_force['MT_STATUS']==mt_status, param].values, weights=weights, rng=rng)

            new_line = {'parameter': param, 
                        'weighted_average': weighted_average,
                        'conf_int_low': ci[0], 
                        'conf_int_high': ci[1],
                        'y_err_low': weighted_average - ci[0],
                        'y_err_high': ci[1] - weighted_average,
                        'MT_STATUS': mt_status
                        }
            df_params_stats = pd.concat([df_params_stats, pd.DataFrame(new_line, index=[0])])

        res = ttest_ind(df_binned_force.loc[df_binned_force['MT_STATUS']==mt_codes[1], param].values, 
                        df_binned_force.loc[df_binned_force['MT_STATUS']==mt_codes[0], param].values, alternative='two-sided', usevar='unequal')
        pval = res[1]
        
        p_values[param] = pval

    df_params_stats.to_csv(f'{dir_plots}/results/results_material_properies_stats_{subtr_bck_label}.csv', index=False)


    df_results_from_average_curves = pd.read_csv(f'{dir_plots}/results/results_parameters_curve_averaged_force_range{force_ranges[0][0]}-{force_ranges[0][1]}pN_{subtr_bck_label}{creep_only_label}.csv')
    df_results_from_average_curves['group_size'] = df_results_from_average_curves .groupby(['EMBRYO', 'MT_STATUS']).size().values

    df_params_avg_curves_stats = pd.DataFrame()

    p_values_avg_curves = dict()

    for param in params:
        if param in df_results_from_average_curves.columns:
            weights_ = set()
            for mt_status in mt_codes:

                weights=df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_status, 'group_size'].values
                weighted_average, ci = weighted_bootstrap(df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_status, param].values, weights=weights, rng=rng)

                new_line = {'parameter': param, 
                            'weighted_average': weighted_average,
                            'conf_int_low': ci[0], 
                            'conf_int_high': ci[1],
                            'y_err_low': weighted_average - ci[0],
                            'y_err_high': ci[1] - weighted_average,
                            'MT_STATUS': mt_status
                            }
                df_params_avg_curves_stats = pd.concat([df_params_avg_curves_stats, pd.DataFrame(new_line, index=[0])])


            weights = (df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[0], 'COUNT'].values*len(df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[0], 'COUNT'].values)/np.sum(df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[0], 'COUNT'].values), 
            df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[1], 'COUNT'].values*len(df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[1], 'COUNT'].values)/np.sum(df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[1], 'COUNT'].values))

            res  = ttest_ind(
                df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[0], param].values, df_results_from_average_curves.loc[df_results_from_average_curves['MT_STATUS']==mt_codes[1], param].values,  weights=weights, alternative='two-sided', usevar='unequal')
        
            pval = res[1]
        
            p_values_avg_curves[param] = pval

    df_params_avg_curves_stats.to_csv(f'{dir_plots}/results/results_material_properies_avg_curves_stats_{subtr_bck_label}.csv', index=False)
    # print('P-values for the parameters from the averaged curves:', p_values_avg_curves)

    df_data_stats = df_data_stats.sort_values(by='MT_STATUS', key=lambda x: x.map(custom_dict))

    for (param, label) in zip(params, labels):
        df = df_results_filtered.sort_values(by='MT_STATUS', key=lambda x: x.map(custom_dict))
        if param in ['k', 'eta_1', 'eta_2', 'elastic_viscous_ratio', 'a', 'tau_r']:
            df = filter_data_params(df_results_filtered).sort_values(by='MT_STATUS', key=lambda x: x.map(custom_dict))
        p_value = p_values[param]
        fig = plt.figure(figsize=(3, 4), dpi=300)
        sns.stripplot(x='MT_STATUS', y=param, hue='MT_STATUS', data=df, jitter=0.4, palette=color_palette, alpha=0.1, linewidth=0.5, legend=False, size=3, zorder=0)

        if param in df_results_from_average_curves.columns:
            p_value_avg_curves = p_values_avg_curves[param]
            sns.stripplot(x='MT_STATUS', y=param, hue='MT_STATUS', data=df_results_from_average_curves, jitter=0.2, alpha=0.8, linewidth=0.5, legend=False, size=4, palette=color_palette, zorder=1)


        plt.errorbar(df_params_avg_curves_stats.loc[df_params_avg_curves_stats['parameter']==param, 'MT_STATUS'], 
                    df_params_avg_curves_stats.loc[df_params_avg_curves_stats['parameter']==param, 'weighted_average'], 
                    yerr=np.array([df_params_avg_curves_stats.loc[df_params_avg_curves_stats['parameter']==param, 'y_err_low'].values, df_params_avg_curves_stats.loc[df_params_avg_curves_stats['parameter']==param, 'y_err_high'].values]), 
                    marker = '_', markersize=40,  color='black', linestyle='', zorder=2)
        print(f'Fold change for {param}:',  df_params_avg_curves_stats.loc[(df_params_avg_curves_stats['parameter']==param)&(df_params_avg_curves_stats['MT_STATUS']==mt_codes[0]), 'weighted_average'].values/df_params_avg_curves_stats.loc[(df_params_avg_curves_stats['parameter']==param)&(df_params_avg_curves_stats['MT_STATUS']==mt_codes[1]), 'weighted_average'].values)

        plt.title(f'p = {p_value_avg_curves:.2g}')
        plt.ylabel(label)
        plt.xlabel('')
        plt.ylim(bottom=0)
        plt.xticks(ticks = mt_codes, labels=conditions)
        if param == 'k':
            plt.ylim(bottom=0, top=150)
        elif param == 'eta_1':
            plt.ylim(bottom=0, top=300)
        elif param == 'eta_2':
            plt.ylim(bottom=0, top=1000)
        elif param == 'elastic_viscous_ratio':
            plt.ylim(bottom=0, top=5)
            plt.hlines(1, -0.5, 1.5, color='gray', linestyle='--', alpha=0.5)
        elif param == 'rising_dif':
            plt.ylim(bottom=0, top=30)

        plt.tight_layout()
        plt.savefig(f'{dir_plots}/results/result_{param}.svg', format='svg')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process magnetic tweezers data and generate extended measurements.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Run the main function with the provided config file
    args = parser.parse_args()
    main(args.config)
    print('All done! :)')