"""
This module provides an engine that utilizes functions for performing exploratory data analysis (EDA) on processed data.
"""
import logging
import traceback
import geopandas as gpd
from utils.common import convert_gpd, read_data, read_equi, read_iris, iris_prep
from eda.utilities import ( create_output_dir, modify_geo_data,
    read_communes, select_equi,
    transform_equi, select_variables
    )
from eda.core import *


def eda_engine(data_path):
    """
    Performs exploratory data analysis (EDA) on the input data.

    Args:
        data_path (str): The path to the processed data file.

    Returns:
        bool: True if the EDA is completed successfully, False otherwise.
    """

    example_area = 'PARIS' #Change to prefered Area
    try:
        # Read processed data
        data = read_data(data_path)
        print('Ready to start EDA')
        print('****************************')
    except FileNotFoundError:
        print("Error: data file not found")
        return False
    try:
        # Create or return directory
        output_dir = create_output_dir()

        # Plot correlations and distribution
        plot_correlation_matrix(data, output_dir)
        plot_heatmap(data, output_dir)

        data = select_variables(data)
        distribution_target_type_and_metropoles(data, output_dir)

        # Plot 'Maison' and 'Appartement' percentage per metropole
        #plot_flats_houses_shares(data, output_dir)

        # Generate box and boxen plots for 'surface_reelle_bati' 
        #and 'nombre_pieces_principales' for each 'type_local'
        #box_flats_houses(data, output_dir)
        #boxen_flats_houses(data, output_dir)

        box_flats_houses_metropoles(data, output_dir)

        # Convert data to geopandas
        geo_data = convert_gpd(data)

        # Read communes and iris
        iris_value, iris_shape = read_iris()
        commune = read_communes()
        amenities = read_equi()


        ## Add information about the IRIS area
        iris = iris_prep(iris_value, iris_shape)
        iris['iris_geometry'] = iris.geometry 
        # Join data
        joined_geo_data = gpd.sjoin(geo_data, iris, how = 'left', op = 'within')
        filtered_data = joined_geo_data.drop(columns = ['iris_geometry'])

        # Modify tables
        data, iris, commune = modify_geo_data(filtered_data, iris, commune)

        ## Plot income and mean price maps
        # Give example with 'Paris' and 'DISP_RD19' variable
        plot_var_iris(iris, example_area, 'DISP_RD19',output_dir)

        #
        #bien_prix_m2(commune, data, 'NICE',output_dir)

        #
        #iris_bien(data, iris, example_area,output_dir)

        #Iris + bien moyen, you can specify metropole and background variable
        # Give example Give example with 'Paris' and 'DISP_RD19' variable
        iris_bien_moyen(data, iris, example_area , metrique = 'prix_m2_actualise',
                         var_iris = 'DISP_EQ19',
                        name_var_iris = 'IQR divided by the mean of incomes', 
                        output_dir = output_dir)
        # Give example with 'Nice' and 'DISP_EQ19' variable                
        iris_bien_moyen(data, iris, 'NICE', metrique = 'prix_m2_actualise', 
                        var_iris = 'DISP_EQ19',
                        name_var_iris = 'IQR divided by the mean of incomes', 
                        output_dir = output_dir)

        # Plot amenities maps
        geo_amenities = convert_gpd(amenities, equi=True)

        amenities = select_equi(geo_amenities)
        amenities = transform_equi(amenities, str(iris.crs))

        plot_equi_commune(amenities, commune, example_area, '75118', output_dir)

        plot_equi_iris(amenities, iris, '751187022', output_dir)

        plot_corr_spatiale(data, iris, commune, example_area, method = 'spearman',
                         output_dir = output_dir )
        print('****************************') 
    except Exception as e:
        logging.error("An error occurred while performing EDA: %s", e)
        print(traceback.format_exc())
        return False  
        
    return True
