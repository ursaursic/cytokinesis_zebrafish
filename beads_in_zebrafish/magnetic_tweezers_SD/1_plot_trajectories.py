import pandas as pd
import os
from tqdm import tqdm  # For displaying progress bars during iteration
import yaml  # For reading YAML configuration files
import argparse  # For parsing command-line arguments

from utils import *  # Assuming utility functions like plot_trajectories and plot_displacement are in this module


def main(config_path):
    """
    Only run if you want to plot trajectories and displacements. 
    Process files, generate plots, and save them based on a configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    
    # Load configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Define necessary directories based on the configuration
    dir_plots = os.path.join(config['dir_parent'], '3_plots')
    dir_analysis = os.path.join(config['dir_parent'], '2_analysis')
    dir_measurements_extended = os.path.join(dir_analysis, 'measurements_extended_info')
    

    # Iterate through all files in the 'measurements_extended_info' directory
    for filename in tqdm(os.listdir(dir_measurements_extended)):
        # Construct the full file path and get the base filename without the extension
        filepath = os.path.join(dir_measurements_extended, filename)
        filename = filename.split('.')[0]  # Extract base name without extension
        
        # Read HDF5 file and retrieve the data and metadata
        df = pd.read_hdf(filepath, key='df')  # Load the dataframe
        with pd.HDFStore(filepath, mode='r') as store:
            comments = store.get_storer('df').attrs.metadata  # Retrieve metadata

        # Create directories for storing plots if they don't already exist
        os.makedirs(os.path.join(dir_plots, 'trajectories'), exist_ok=True)
        os.makedirs(os.path.join(dir_plots, 'displacement'), exist_ok=True)

        # Save plots as PNG files in the appropriate directories
        plot_trajectories(filename, df, comments, save_to_filepath=os.path.join(dir_plots, 'trajectories', f'{filename}.png'), show_background_fit=True)
        plot_displacement(filename, df, comments, save_to_filepath=os.path.join(dir_plots, 'displacement', f'{filename}.png'))


if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    
    # Parse the arguments and execute the main function
    args = parser.parse_args()
    main(args.config)
    print("All done! :)")
