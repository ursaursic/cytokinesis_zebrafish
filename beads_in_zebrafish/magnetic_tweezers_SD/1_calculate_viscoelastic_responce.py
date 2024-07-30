
import pandas as pd
import os
from tqdm import tqdm
import yaml
import argparse

from utils import *


def main(config_path):
    # load config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dir_plots = config['dir_parent']+'/3_plots'
    dir_analysis = config['dir_parent']+'/2_analysis'
    dir_measurements_extended = f'{dir_analysis}/measurements_extended_info'
    subtract_background = config['subtract_background']
    ################################################################################
    # save_mode = 'ipynb' # the code will display the plots in browser 
    save_mode = 'png' # the plots will be saved as png on fileserver
    filepath_results_corrected = f'{dir_analysis}/results_viscoelasticity_corrected.csv' # result filepath for data with background flow correction (use this!)
    filepath_results = f'{dir_analysis}/results_viscoelasticity.csv' # result filepath for data without background correction
    ################################################################################

    # Results data frame
    df_results = pd.DataFrame(columns=['FILENAME', 'TRACK_IDX', 'PULSE_NUMBER', 'MT_STATUS', 'VISCOEL_PARAMS_RISING', 'VISCOEL_PARAMS_RELAXING', 'COMMENTS'], index=[0])

    for filename in tqdm(os.listdir(dir_measurements_extended)):
        filepath = dir_measurements_extended+'/'+filename
        filename = filename.split('.')[0]  # This is the name of the file without extention
        # Open the HDF5 file and read metadata
        df = pd.read_hdf(filepath, key='df')
        with pd.HDFStore(filepath, mode='r') as store:
            comments = store.get_storer('df').attrs.metadata
        
        try:
            df = add_calculated_displacement(df, subtract_background=subtract_background) # calculate displacement for every pulse (option: substract background flows)
        except:
            continue

        intermediate_plots=True
        if intermediate_plots:
            if not os.path.exists(f'{dir_plots}/trajectories'):
                os.mkdir(f'{dir_plots}/trajectories')
            if not os.path.exists(f'{dir_plots}/displacement'):
                os.mkdir(f'{dir_plots}/displacement')
            
            if save_mode == 'ipynb':
                plot_trajectories(filename, df, comments, save_to_filepath='ipynb', show_background_fit=True)
                plot_displacement(filename, df, comments, save_to_filepath='ipynb')
            elif save_mode == 'png':
                plot_trajectories(filename, df, comments, save_to_filepath=f'{dir_plots}/trajectories/{filename}.png', show_background_fit=True)
                plot_displacement(filename, df, comments, save_to_filepath=f'{dir_plots}/displacement/{filename}.png')
            else:
                print('Intermendiate plots not saved.')

        if save_mode == 'ipynb':
            df_results = calculate_fit_parameters(filename, df, comments, df_results, save_plot_to_filepath='ipynb')
        elif save_mode == 'png':
            df_results = calculate_fit_parameters(filename, df, comments, df_results, save_plot_to_filepath=f'{dir_plots}')
        else:
            print('Displacement-force curves not saved.')


    if subtract_background:
        # subtracted background flows (this data is more accurate)
        df_results.to_csv(filepath_results_corrected)
    else:
        # original data without background flow subtraction
        df_results.to_csv(filepath_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    main(args.config)
    print("All done! :)")
