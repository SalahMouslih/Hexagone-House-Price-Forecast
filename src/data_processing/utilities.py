import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from sklearn.linear_model import LinearRegression
import glob
import os
from joblib import Parallel, delayed


# Import tables 
path_to_metropole = 'data/open_data/metropoles_communes.csv'
metropoles = pd.read_csv(path_to_metropole, delimiter=';', header=5)
iris_value = pd.read_csv('data/open_data/IRIS_donnees.csv', delimiter = ';')
iris_shape = gpd.read_file('data/open_data/IRIS_contours.shp')


def read_dvfs(data_paths):
    """
    Read multiple DVF data from the given paths and return a single concatenated dataframe.
    """
    try:
        print('Reading data...')

        # Use glob to find all csv files matching the data paths
        data_paths = [path for pattern in data_paths for path in glob.glob(pattern)]
        
        # Print the list of file paths for debugging purposes
        data = pd.concat(map(pd.read_csv, data_paths))

        print('Ready to start preprocessing')
        print('************************')

        return data
    
    except Exception as e:
        print(f"Error occurred while reading data: {e}")
        return None


def read_tables(*data_paths):
    """
    Read multiple csv files from the given paths and return a list of dataframes.
    """
    dfs = [] 

    # iterate over each path in the input arguments
    for path in data_paths: 
        df = pd.read_csv(path) 
        # add the resulting dataframe to the list
        dfs.append(df) 

    # return the list of dataframes
    return dfs 


def get_top_zones(df, nb_top_zones):

    """Select zones where the highest number mutations"""

    try:
        print('Selecting top 10 metropoles...')

        # Correct the spelling of regions
        df.loc[df.nom_commune.str.startswith('Marseille '), 'nom_commune'] = 'Marseille'
        df.loc[df.nom_commune.str.startswith('Lyon '), 'nom_commune'] = 'Lyon'
        df.loc[df.nom_commune.str.startswith('Paris '), 'nom_commune'] = 'Paris'

        # Merge dvf and metropole
        df = df.merge(metropoles, how='left', left_on='nom_commune', right_on='LIBGEO')

        # Pick the areas with the highest number of transactions
        most_frequent = df['LIBEPCI'].value_counts().head(nb_top_zones).index.to_list()
        df = df.loc[df['LIBEPCI'].isin(most_frequent)]

        return df

    except Exception as e:
        print(f"An error occurred while selecting the top {nb_top_zones} zones: {str(e)}")
        return None


def convert_gpd(df):
    """
    Function convert_gpd converts a pandas DataFrame to a GeoDataFrame using the geometry attribute 
    which is created from the longitude and latitude columns of the input DataFrame
    """
    return gpd.GeoDataFrame(
        df, geometry = gpd.points_from_xy(df.longitude, df.latitude)
        )


def get_k_nearest_neighbors(source_points, candidate_points, k_neighbors):
    """Find the k nearest neighbors for all source points from a set of candidate points"""
    tree = BallTree(candidate_points, leaf_size=15, metric='haversine')
    distances, indices = tree.query(source_points, k=k_neighbors)
    return indices, distances


def get_nearest_neighbors(left_gdf, right_gdf, k_neighbors, return_distances=False):
    """
    For each point in left_gdf, find the k-nearest neighbors in right_gdf and return their indices.
    Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    left_geom_col_name = left_gdf.geometry.name
    right_geom_col_name = right_gdf.geometry.name

    #ensure that index in right gdf is formed of sequential numbers
    right_gdf = right_gdf.reset_index(drop=True)

    # convert coordinates to radians
    left_radians_x = left_gdf[left_geom_col_name].x.apply(lambda geom: geom * np.pi / 180)
    left_radians_y = left_gdf[left_geom_col_name].y.apply(lambda geom: geom * np.pi / 180)
    left_radians = np.c_[left_radians_x, left_radians_y]

    right_radians_x = right_gdf[right_geom_col_name].x.apply(lambda geom: geom * np.pi / 180)
    right_radians_y = right_gdf[right_geom_col_name].y.apply(lambda geom: geom * np.pi / 180)
    right_radians = np.c_[right_radians_x, right_radians_y]

    indices, distances = get_k_nearest_neighbors(source_points=left_radians,
                                                 candidate_points=right_radians,
                                                 k_neighbors=k_neighbors)
    if return_distances:
        return indices, distances
    else:
        return indices


def apply_linear_regression(row, metric_of_interest):

    """Apply linear regression to calculate the intercept of a row with the given metric of interest."""
    indices = row['indices']
    X = table_info.loc[indices, ['surface_reelle_bati', 'nombre_pieces_principales']].values
    y = table_info.loc[indices, metric_of_interest].values

    lr = LinearRegression()
    lr.fit(X, y)

    return lr.intercept_


def calculate_closest_metric(dvf, table_info, k_neighbors, metric_of_interest, new_metric_name, apply_regression=False):
    """Compute the new metric based on the k-nearest neighbors in table_info dataframe."""
    try:
        print(f"Computing {new_metric_name}...")
        dvf[new_metric_name] = np.nan
        closest_indices = get_nearest_neighbors(left_gdf=dvf, right_gdf=table_info, k_neighbors=k_neighbors)
        dvf['indices'] = list(closest_indices)

        if apply_regression: 
            dvf[new_metric_name] = dvf.swifter.apply(lambda row: apply_linear_regression(row, metric_of_interest), axis=1)
        else:
            dvf[new_metric_name] = dvf['indices'].apply(lambda indices: table_info[metric_of_interest].iloc[indices].mean())

        return dvf

    except Exception as e:
        print("Error: could not calculate closest metric")
        print(str(e))
        return None


def iris_prep():
    """
    Merge iris_shape and iris_value tables to obtain the polygons and the IRIS values in the same table.

    Parameters: None
    Returns: A pandas dataframe containing the merged iris data with no duplicate entries based on 'DCOMIRIS' column.
    """
    try:
        # Remove duplicates from iris_shape and iris_value tables
        iris_shape = iris_shape.drop_duplicates(subset=['DCOMIRIS'], keep='first')
        iris_value = iris_value.drop_duplicates(subset=['IRIS'], keep='first')

        # Convert 'IRIS' column to a string of 9 characters with leading zeros if necessary
        iris_value['IRIS'] = iris_value['IRIS'].astype(str).str.rjust(9, '0')

        # Merge iris_shape and iris_value tables and remove duplicates based on 'DCOMIRIS' column
        iris = iris_shape.merge(iris_value, how='left', right_on='IRIS', left_on='DCOMIRIS')
        iris = iris.drop_duplicates(subset=['DCOMIRIS'], keep='first')

        return iris
    except KeyError as e:
        print(f"Error: {str(e)} column not found in input data")
    except Exception as e:
        print(f"Error: {str(e)}")


income_input_variable_names = ['DISP_TP6019', 'DISP_Q119', 'DISP_MED19', 'DISP_Q319', 'DISP_EQ19', 'DISP_D119', 'DISP_D219',
                        'DISP_D319', 'DISP_D419', 'DISP_D619', 'DISP_D719', 'DISP_D819', 'DISP_D919', 'DISP_RD19',
                        'DISP_S80S2019', 'DISP_GI19', 'DISP_PACT19', 'DISP_PTSA19', 'DISP_PCHO19', 'DISP_PBEN19',
                        'DISP_PPEN19', 'DISP_PPAT19', 'DISP_PPSOC19', 'DISP_PPFAM19', 'DISP_PPMINI19', 'DISP_PPLOGT19',
                        'DISP_PIMPOT19', 'DISP_NOTE19']
income_output_variable_names = ['Taux_pauvreté_seuil_60', 'Q1', 'Mediane', 'Q3', 'Ecart_inter_Q_rapporte_a_la_mediane', 'D1',
                         'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'Rapport_interdécile_D9/D1', 'S80/S20', 'Gini',
                         'Part_revenus_activite', 'Part_salaire', 'Part_revenus_chomage', 'Part_revenus_non_salariées',
                         'Part_retraites', 'Part_revenus_patrimoine', 'Part_prestations_sociales',
                         'Part_prestations_familiales', 'Part_minima_sociaux', 'Part_prestations_logement', 'Part_impôts']

equi_input_variable_names=['A203', 'A206', 'B101', 'C101', 'C201', 'D201', 'E107', 'F303', 'F307', 'F313']

equi_output_variable_names = ['Banques', 'Bureaux_de_Poste', 'Commerces', 'Ecoles','Collèges_Lycées', 'Medecins','Gares', 'Cinema',
                        'Bibliotheques', 'Espaces_remarquables_et_patrimoine']            


def choose_metric_names(df, variable):
    """
    Calculates a new metric using the given input metric and name.
    """
    if variable == 'income':
        return alter_metric_name(df,income_input_variable_names, income_output_variable_names) 
    elif variable == 'equip':
        return alter_metric_name(df,equi_input_variable_names, equi_output_variable_names) 


def alter_metric_name(df,input_variable_names,output_variable_names):
    """
    Calculate new metrics using my_choose_closest() function and return updated dataframe.

    Parameters:
    df (pandas dataframe): dataframe to calculate new metrics on.
    input_variable_names (list): names of variables to calculate new metrics from.
    output_variable_names (list): names to give new metrics.

    Returns:
    df (pandas dataframe): updated dataframe with input variables dropped and new metrics added.
    """
    # Define a helper function to calculate a single new metric using my_choose_closest() function
    def calculate_single_metric(input_var, output_var):
        return calculate_closest_metric(dvf=dvf_geo,
                                        table_info=dvf_geo[dvf_geo[input_metric].notnull()], 
                                        k_neighbors=1,
                                        metric_of_interest=input_metric, 
                                        new_metric_name=new_metric_name)[new_metric_name]
    
    # Calculate the new metrics using parallel processing
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(calculate_single_metric)(input_var, output_var) for input_var, output_var in zip(input_variable_names, output_variable_names))
    
    # Create a dictionary of new metric names and values
    new_metrics_dict = {output_var: result for output_var, result in zip(output_variable_names, results)}
    
    # Add the new metrics to the df dataframe
    df = df.assign(**new_metrics_dict)
    
    # Drop the input variables from the df dataframe
    df = df.drop(columns=input_variable_names)
    
    return df

liste_var_garder=['id_mutation', 'date_mutation', 'numero_disposition', 'valeur_fonciere',
       'adresse_numero', 'adresse_nom_voie', 'adresse_code_voie',
       'code_commune', 'nom_commune', 'code_departement', 'LIBEPCI',
       'id_parcelle', 'nombre_lots', 'lot1_numero', 'lot1_surface_carrez',
       'lot2_numero', 'lot2_surface_carrez', 'lot3_numero',
       'lot3_surface_carrez', 'lot4_numero', 'lot4_surface_carrez',
       'lot5_numero', 'lot5_surface_carrez', 'type_local',
       'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain',
       'longitude', 'latitude', 'geometry', 'quantile_prix', 'coeff_actu','prix_actualise','prix_m2_actualise','prix_m2','trimestre_vente','prix_m2_zone',
        'moyenne','moyenne_brevet','DCOMIRIS','indices', 'Banques', 'Bureaux_de_Poste', 'Commerces', 'Ecoles','Collèges_Lycées', 'Medecins',
       'Gares', 'Cinema', 'Bibliotheques', 'Espaces_remarquables_et_patrimoine', 'DCIRIS',
       'Taux_pauvreté_seuil_60', 'Q1', 'Mediane', 'Q3', 'Ecart_inter_Q_rapporte_a_la_mediane', 'D1', 'D2', 'D3', 'D4',
       'D5', 'D6', 'D7', 'D8', 'D9', 'Rapport_interdécile_D9/D1', 'S80/S20', 'Gini', 'Part_revenus_activite',
       'Part_salaire', 'Part_revenus_chomage', 'Part_revenus_non_salariées', 'Part_retraites', 'Part_revenus_patrimoine',
       'Part_prestations_sociales', 'Part_prestations_familiales', 'Part_minima_sociaux', 'Part_prestations_logement','Part_impôts']

def select_variables(dvf_geo, keep_columns = liste_var_garder):
    """
    Select variables from dvf_geo dataframe and return updated dataframe.

    Parameters:
    dvf_geo (pandas dataframe): dataframe to select variables from.
    liste_var_garder (list): list of variables to keep in the updated dataframe.

    Returns:
    dvf_geo_final (pandas dataframe): updated dataframe with selected variables.
    """
    dvf_geo_final = dvf_geo[keep_columns]
    
    return dvf_geo_final
