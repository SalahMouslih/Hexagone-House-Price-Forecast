"""
This module provides an engine that utilizes functions for performing exploratory data analysis (EDA) on processed data.
"""
from utils.common import read_data, read_iris, iris_prep
from eda.core import *
from eda.utilities import create_output_dir, select_variables


def eda_engine(data_path):
    """
    Performs exploratory data analysis (EDA) on the input data.

    Args:
        data_path (str): The path to the processed data file.

    Returns:
        bool: True if the EDA is completed successfully, False otherwise.
    """
    try:
        # Read processed data
        data = read_data(data_path)
        print('Ready to start EDA')
        print('****************************')
    except FileNotFoundError:
        print("Error: data file not found")

    # Create or return directory
    output_dir = create_output_dir()

    # Plot correlations and distribution
    plot_correlation_matrix(data, output_dir)
    plot_heatmap(data, output_dir)

    data = select_variables(data)
    distribution_target_type_and_metropoles(data, output_dir)

    # Plot 'Maison' and 'Appartement' percentage per metropole
    plot_flats_houses_shares(data, output_dir)

    # Generate box and boxen plots for 'surface_reelle_bati' and 'nombre_pieces_principales' for each 'type_local'
    box_flats_houses(data, output_dir)
    boxen_flats_houses(data, output_dir)

    box_flats_houses_metropoles(data, output_dir)
    print('****************************')

    return True