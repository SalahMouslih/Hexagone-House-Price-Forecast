"""
This module contains functions for filtering and selecting data from a given dataset.

Functions:

    -select_bien(df): Filter the dataset to keep only properties of type 'Maison' or 'Appartement' that are being sold.
    -filtre_dur(df, bati, piece, local, metropole_name=None): Filter the dataset to keep only properties of a given type,
    within or outside a given metropolitan area, and within given building surface and number of rooms constraints.
    -filtre_prix(df, metric_prix, quantile_nv = 0.99): Filter the dataset to keep only properties with a price per square 
    meter below the 99th percentile for each city and property type.
"""
import pandas as pd
import numpy as np


def select_bien(df):  
    """
    Filter and select specific property types from a given DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing property transaction data.

    Returns:
        pandas.DataFrame: Filtered DataFrame containing only property transactions that are
        of type 'Vente', are either 'Maison' or 'Appartement', and have known
        latitude and longitude values.Returns None if KeyError or TypeError occurs.
    """  
    try:
        print("Filtering property types...")

        # Filter by transacation type
        df = df[df['nature_mutation'] == 'Vente']
        # Filter by property type. Keep only the 'Maison' and 'Appartement' properties
        df = df.loc[df['type_local'].isin(['Maison', 'Appartement'])]
        # Keep only properties with known locations
        # our analysis heavily relies on property location
        df = df[(df['latitude'].notna()) & (df['longitude'].notna())]

        return df

    except KeyError as ke:
        print(f"KeyError: {ke}")
        return None 
    except TypeError as te:
        print(f"TypeError: {te}")
        return None

def filtre_dur(df, bati, piece, local, metropole_name=None):
    """
    Filter out outlier values for a given metropolitan area and property type.

    Args:
        df (pd.DataFrame): Input dataset.
        bati (int): Maximum allowed building surface.
        piece (int): Maximum allowed number of rooms.
        local (str): Type of property ('Maison' or 'Appartement').
        metropole_name (str, optional): Name of the metropolitan area to be filtered.

    Returns:
        pd.DataFrame: The filtered dataset.    
    """
    try:
        # Filter data for the given local in metropole
        if metropole_name:
            print(f"Filtering data for '{local}' in '{metropole_name}'...")

            df_metropole = df[(df['type_local'] == local) & (df['LIBEPCI'] == metropole_name)]
            df_other_metropoles = df[(df['LIBEPCI'] != metropole_name) | ((df['LIBEPCI'] == metropole_name) & (df['type_local'] != local))]
        else:
            # Filter data for the given local across all metropoles
            print(f"Filtering data for '{local}'")
            df_metropole = df[df['type_local'] == local]
            df_other_metropoles = df[df['type_local'] != local]

        # Filter data based on given 'bati' and 'piece' constraints
        df_metropole = df_metropole[(df_metropole['surface_reelle_bati'] <= bati) &
                                    (df_metropole['nombre_pieces_principales'] <= piece)]

        # merge filtered data for the given local in metropole with data for other metropoles
        filtered_df = pd.concat([df_metropole, df_other_metropoles])

        return filtered_df    
    except Exception as e:
        print(f"Error occurred in filtre_dur(): {str(e)}")
        return None

def filtre_prix(df, metric_prix, quantile_nv = 0.99):
    """ 
    Compute the 99th percentile for each city (more precise than EPCI) and property 
    type (Appartement, Maison).Filter properties based on their price per square meter 
    being below the 99th percentile

    ++++++ Be careful to use the discounted price ++++++

    Args:
        df (pd.DataFrame): Input dataset.
        metric_prix (str): Name of the column with the price data.
        quantile_nv (float, optional): The quantile value to compute (default is 0.99).

    Returns:
        pd.DataFrame: The filtered dataset.

    """
    try:
        print('Filtering prices...')

        # Remove properties with prices below 1000 euros per square meter or above 20000 euros 
        #per square meter
        df = df[(df[metric_prix] >= 1000) & (df[metric_prix] <= 20000)]

        # Compute the 99th percentile for each city and property type
        quantile_per_city_type = (
                df.groupby(['nom_commune', 'type_local'])
                .agg({metric_prix: lambda x: np.quantile(x, quantile_nv)})
                .reset_index()
                .rename(columns={metric_prix: 'quantile_prix'})
            )

        # Merge the 99th percentile values with the original DataFrame
        df = df.merge(quantile_per_city_type, on=['nom_commune', 'type_local'], how='left')
        # Filter out properties with prices per square meter above the 99th percentile
        filterd_df = df[df[metric_prix] < df['quantile_prix']]

        return filterd_df

    except Exception as e:
        print(f"Error in filtre_prix: {str(e)}")
        return None
