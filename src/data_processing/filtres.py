import pandas as pd


def select_bien(df):

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
    
    if metropole_name:
        df_metropole = df[(df['type_local'] == local) & (df['LIBEPCI'] == metropole_name)]
        df_other_metropoles = df[(df['LIBEPCI'] != metropole_name) | ((df['LIBEPCI'] == metropole_name) & (df['type_local'] != local))]
    else:
        df_metropole = df[df['type_local'] == local]
        df_other_metropoles = df[df['type_local'] != local]
    
    df_metropole = df_metropole[(df_metropole['surface_reelle_bati'] <= bati) &
                                (df_metropole['nombre_pieces_principales'] <= piece)]
    
    # merge filtered data for the given local in metropole with data for other metropoles
    df_filtered = pd.concat([df_metropole, df_other_metropoles])
    
    return df_filtered

def filtre_prix(df, metric_prix, quantile_nv = 0.99):
    """ 
    Compute the 99th percentile for each city (more precise than EPCI) and property type (Appartement, Maison)
    Filter properties based on their price per square meter being below the 99th percentile

    Parameters:
    df (pd.DataFrame): Input dataset.
    metric_prix (str): Column name for price metric.
    quantile_nv (float, optional): Quantile value. Default is 0.99.

    ++++++ Be careful to use the actualised price ++++++

    """
    df = df[(df[metric_prix] >= 1000) & (df[metric_prix] <= 20000)]

    quantile_per_city_type = (
            df.groupby(['nom_commune', 'type_local'])
            .agg({metric_prix: lambda x: np.quantile(x, quantile_nv)})
            .rename(columns={metric_prix: 'quantile_prix'})
            .reset_index()
        )

    df = df.merge(quantile_per_city_type, on=['nom_commune', 'type_local'], how='left')
    df = df[df[metric_prix] < df['quantile_prix']]

    return df
