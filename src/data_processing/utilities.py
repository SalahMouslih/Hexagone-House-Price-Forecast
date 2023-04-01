import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.linear_model import LinearRegression
import glob
import os


# Import the table which defines the metrpole (EPCI)

#enter path
#metropoles = pd.read_csv(path, delimiter=';', header=5)

def read_dvfs(data_paths):
    print('Reading data...')

    # Use glob to find all csv files matching the data paths
    data_paths = [path for pattern in data_paths for path in glob.glob(pattern)]
    
    # Print the list of file paths for debugging purposes
    data = pd.concat(map(pd.read_csv, data_paths))
    return data


def get_top_zones(df, nb_top_zones):

    """Select zones where the highest number mutations"""

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


def convert_gpd(df):
    return gpd.GeoDataFrame(
        df, geometry = gpd.points_from_xy(df.longitude, df.latitude))


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


def calculate_closest_metric(dvf, table_info, k_neighbors, metric_of_interest, new_metric_name, apply_regression=False):
    """Compute the new metric based on the k-nearest neighbors in table_info dataframe."""
    dvf[new_metric_name] = np.nan
    closest_indices = get_nearest_neighbors(left_gdf=dvf, right_gdf=table_info, k_neighbors=k_neighbors)

    if apply_regression:
        def apply_linear_regression(row, metric_of_interest):
            indices = row['indices']
            X = table_info.loc[indices, ['surface_reelle_bati', 'nombre_pieces_principales']].values
            y = table_info.loc[indices, metric_of_interest].values

            lr = LinearRegression()
            lr.fit(X, y)
    
            return lr.intercept_
        dvf[new_metric_name] = dvf.swifter.apply(lambda row: apply_linear_regression(row, metric_of_interest), axis=1)
    else:
        dvf[new_metric_name] = dvf['indices'].apply(lambda indices: table_info[metric_of_interest].iloc[indices].mean())

    return dvf

