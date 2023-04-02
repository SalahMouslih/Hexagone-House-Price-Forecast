import pandas as pd
import numpy as np

def select_bien(df):
    
    print("Filtereing property types...")

    # Keep 'Ventes' transactions
    df = df[df['nature_mutation'] == 'Vente']
    # Keep the 'Maison' and 'Appartement' properties
    df = df.loc[df['type_local'].isin(['Maison', 'Appartement'])]
    # Keep only properties with known locations
    # our analysis heavily relies on property location
    df = df[(df['latitude'].notna()) & (df['longitude'].notna())]

    return df


def filtre_dur(df, bati, piece, local, metropole_name=None):
    """
    Filter out outlier values for a given metropolitan area and property type.

    Parameters:
    df (pd.DataFrame): Input dataset.
    bati (int): Maximum allowed building surface.
    piece (int): Maximum allowed number of rooms.
    local (str): Type of property ('Maison' or 'Appartement').
    metropole_name (str, optional): Name of the metropolitan area to be filtered.
    """
    # if a metropole name is given, filter data for the given local in that metropole
    if metropole_name:
        print(f"Filtereing data for '{local}' in ' {metropole_name}'...")

        df_metropole = df[(df['type_local'] == local) & (df['LIBEPCI'] == metropole_name)]
        df_other_metropoles = df[(df['LIBEPCI'] != metropole_name) | ((df['LIBEPCI'] == metropole_name) & (df['type_local'] != local))]
    else:
        # if no metropole name is given, filter data for the given local across all metropoles
        print(f"Filtereing data for '{local}'")
        df_metropole = df[df['type_local'] == local]
        df_other_metropoles = df[df['type_local'] != local]

    # filter data based on given bati and piece constraints
    df_metropole = df_metropole[(df_metropole['surface_reelle_bati'] <= bati) &
                                (df_metropole['nombre_pieces_principales'] <= piece)]
    
    # merge filtered data for the given local in metropole with data for other metropoles
    df_filtered = pd.concat([df_metropole, df_other_metropoles])
    
    return df_filtered


def filtre_prix(df, metric_prix, quantile_nv = 0.99):

    """ 
    Compute the 99th percentile for each city (more precise than EPCI) and property type (Appartement, Maison)
    Filter properties based on their price per square meter being below the 99th percentile

    ++++++ Be careful to use the discounted price ++++++

    """

    print('Filtering prices...')

    # Remove properties with prices below 1000 euros per square meter or above 20000 euros per square meter
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
    df = df[df[metric_prix] < df['quantile_prix']]

    return df