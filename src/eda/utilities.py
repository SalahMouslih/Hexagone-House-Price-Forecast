"""
This module provides utility functions for performing exploratory data analysis (EDA)
on processed data from a single CSV file. It includes functions for reading the
processed data, creating an output directory, selecting variables from a
dataframe, and getting unique values for metropoles.
"""
import pandas as pd
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


def read_processed_data(data_path):
    """
    Read processed data from a single CSV file and return the dataframe.

    Args:
        data_path (str): Path to the processed CSV file.

    Returns:
        pd.DataFrame: Dataframe containing the processed data.
            Returns None if an error occurs while reading the data.
    """
    try:
        print('Reading processed data...')
        data = pd.read_csv(data_path)
        print('Ready to start EDA')
        print('****************************')
        return data[:1000]
    except Exception as e:
        print(f"Error occurred while reading data: {e}")
        return None


def create_output_dir():
     """
    Creates then returns a directory named "data/plots" if it does not exist already.

    Returns:
        str: Path of the output directory.
    """
    # Define the path of the output directory
    output_dir = "data/plots/"
    
    # Create the output directory if it does not exist already
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Return the path of the output directory
    return output_dir

def get_metropoles(data):
    """
    Returns a list of metropoles of the given dataframe.

    Args:
        data (pd.DataFrame): The input dataframe.

    Returns:
        list: A list of metropoles.

    Raises:
        TypeError: If 'data' is not a pandas DataFrame.
        KeyError: If the 'LIBEPCI' column is not present in the dataframe.
        Exception: For any other errors that may occur.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")

        # Extract unique 'LIBEPCI' values
        metropoles = data['LIBEPCI'].unique()
        return metropoles

    except KeyError as e:
        print(f"Error occurred while extracting 'LIBEPCI' values: {e}")
        return None

    except TypeError as e:
        print(f"Error occurred while processing data: {e}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def select_variables(df, keep_columns = liste):
    """
    Select variables from dataframe and return updated dataframe.

    Parameters:
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
