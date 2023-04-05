"""
This module provides utility functions for performing exploratory data analysis (EDA)
on processed data from a single CSV file. It includes functions for creating an output directory, 
selecting variables from a dataframes.
"""
import pandas as pd
import geopandas as gpd
import os


liste=['prix_m2_actualise',
       'nom_commune', 'LIBEPCI', 'code_departement', 'latitude', 'longitude',
       'type_local',
       'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain',
       'prix_m2_zone',
       'trimestre_vente',
       'moyenne', 'moyenne_brevet',
       'Banques', 'Bureaux_de_Poste', 'Commerces', 'Ecoles', 'Collèges_Lycées', 'Medecins', 'Gares', 'Cinema', 'Bibliotheques', 'Espaces_remarquables_et_patrimoine', 
       'Taux_pauvreté_seuil_60', 'Q1', 'Mediane', 'Q3', 'Ecart_inter_Q_rapporte_a_la_mediane',
       'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
       'Rapport_interdécile_D9/D1', 'S80/S20', 'Gini',
       'Part_revenus_activite', 'Part_salaire', 'Part_revenus_chomage', 'Part_revenus_non_salariées', 'Part_retraites', 'Part_revenus_patrimoine',
       'Part_prestations_sociales', 'Part_prestations_familiales', 'Part_minima_sociaux', 'Part_prestations_logement', 'Part_impôts']

def create_output_dir():
    """
    Creates then returns a directory named "data/plots" if it does not exist already.

    Returns:
        str: Path of the output directory.
    """
    try:
        # Define the path of the output directory
        output_dir = "data/output/plots/"
        
        # Create the output directory if it does not exist already
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Return the path of the output directory
        return output_dir
    
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return None


def select_variables(df, keep_columns = liste):
    """
    Select variables from dataframe and return updated dataframe.

    Args:
        df (pandas dataframe): dataframe to select variables from.
        keep_columns (list): list of variables to keep in the updated dataframe.

    Returns:
        df_final (pandas dataframe): updated dataframe with selected variables.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        
        print("Keeping variables of interest...")
        # Keep columns of interest 
        df_final = df[keep_columns]
        return df_final

    except KeyError as e:
        print(f"Error occurred while selecting variables: {e}")
        return None

    except TypeError as e:
        print(f"Error occurred while filtering data: {e}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

liste_equipements = ['A203', 'A206', 'B101', 'C101',  'C201', 'D201', 'E107', 'F303', 'F307',  'F313']

def select_equi(equipements, liste_equi = liste_equipements):
    '''
    Filters the equipements dataframe to select rows whose TYPEQU column is in the given list.

    Args:
        equipements (pandas.DataFrame): a dataframe containing information about equipements in different IRIS
        liste_equi (list): a list of strings containing the names of the equipements to select

    Returns:
        equipements (pandas.DataFrame): the filtered dataframe
    '''

    try:
        equipements = equipements[equipements['TYPEQU'].isin(liste_equi)]
        equipements = equipements.replace({'A203': 'Banque', 'A206': 'Poste', 'B101': 'Alimentation', 'C101': 'Ecoles',  'C201': 'College-Lycee',
                         'D201': 'Medecin', 'E107': 'Gare',  'F303': 'Cinema', 'F307': 'Bibliotheque',  'F313': 'Patrimoine'})
        return equipements

    except Exception as e:
        print(f"Error occurred while selecting equipements: {e}")
        return None


def transform_equi(equipements, crs):
    '''
    Transforms the equipements dataframe to match the given crs.

    Args:
        equipements (pandas.DataFrame): a dataframe containing information about equipements in different IRIS
        crs (str): the coordinate reference system to transform the dataframe to

    Returns:
        equipements_crs (geopandas.GeoDataFrame): the transformed dataframe
    '''

    try:
        if equipements.crs is None:
            equipements = equipements.set_crs(epsg=2154) # sets the initial crs of the dataframe
        equipements_crs = equipements.to_crs(crs) # transforms the dataframe to the given crs
        return equipements_crs

    except Exception as e:
        print(f"Error occurred while transforming equipements: {e}")
        return None


    
def read_communes():
    """
    Read the shapefile containing the commune data and return a GeoDataFrame.
    
    Returns:
        communes (GeoDataFrame): A GeoDataFrame containing the commune data.

    Raises:
        FileNotFoundError: If the data file is not found.
        Exception: If there is an error while reading the data file.
    """

    # Set the path of the shapefile containing the commune data
    communes_shape_path = 'data/open_data/communes-20220101.shp'

    try:
        # Read the data from the shapefile and create a GeoDataFrame
        print("Reading 'communes' tables...")
        communes = gpd.read_file(communes_shape_path)

        # Return the GeoDataFrame
        return communes

    except FileNotFoundError as e:
        # If the data file is not found, print an error message and return None
        print(f"Error occurred while reading data: {e}")
        return None

    except Exception as e:
        # If there is an error while reading the data file, print an error message and return None
        print(f"An error occurred: {e}")
        return None


def modify_geo_data(data, iris, commune):
    """
    Modify columns of communes, data, and iris dataframes.

    Args:
        iris (geoPandas): iris dataset.
        data (geoPandas): data dataset.
        commune (geoPandas): commune dataset.

    Returns:
        tuple: A tuple containing preprocessed iris, data and communes dataframes.


    Raises:
        FileNotFoundError: If commune the input file paths is incorrect.
    """
    try:
        print('Modify geodataframes')
       
        # Alter data columns
        data = data.set_crs(4171)
        data['nom_commune'] = data['nom_commune'].str.upper()
        data.loc[data.NOM_COM.str.startswith('PARIS ').fillna(False), 'NOM_COM'] = 'Paris'
        data.loc[data.NOM_COM.str.startswith('MARSEILLE ').fillna(False), 'NOM_COM'] = 'Marseille'
        data.loc[data.NOM_COM.str.startswith('LYON ').fillna(False), 'NOM_COM'] = 'Lyon'
        data['NOM_COM'] = data['NOM_COM'].str.upper()
        # Alter iris columns
        iris.loc[iris.NOM_COM.str.startswith('PARIS ').fillna(False), 'NOM_COM'] = 'Paris'
        iris.loc[iris.NOM_COM.str.startswith('MARSEILLE ').fillna(False), 'NOM_COM'] = 'Marseille'
        iris.loc[iris.NOM_COM.str.startswith('LYON ').fillna(False), 'NOM_COM'] = 'Lyon'
        iris['NOM_COM'] = iris['NOM_COM'].str.upper()
        
        # Process communes file
        commune.nom = commune.nom.str.upper()

        return data, iris, commune
    except Exception as e:
        print(f"Error occurred while cleaning data: {e}")
        return None
