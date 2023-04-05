"""
This module provides functions to preprocess data on schools and merge it with DVF data.

Functions:
- prep_lyc(data: pd.DataFrame, geo_etab: pd.DataFrame) -> gpd.GeoDataFrame
    Filters the given lycée data to only include lycées généraux, calculates the taux de mention for
    each lycée and converts the result to a geopandas dataframe, which is then merged with
    the dvf data.

- prep_brevet(data, geo_etab)
    Preprocesses brevet data by computing the taux de mention for each college,
    converting it to a geopandas dataframe, and merging it with the DVF dataframe.

"""

import pandas as pd
import geopandas as gpd


def prep_lyc(data: pd.DataFrame, geo_etab: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Filters the given lycée data to only include lycées généraux, as they are more likely to
    influence housing prices than other types of schools. Calculates the taux de mention for
    each lycée and converts the result to a geopandas dataframe, which is then merged with
    the dvf data.

    Args:
    data (pd.DataFrame): a pandas DataFrame containing data on lycées
    geo_etab (pd.DataFrame): a pandas DataFrame containing geographical data on the lycées

    Returns:
    A geopandas GeoDataFrame with the filtered and processed lycée data
    """    
    try:
        # Start by filtering out the data for years other than 2020 and keeping only lycées généraux
        lyc = data[data['Annee'] == 2020]
        lyc_gen = lyc[['Etablissement', 'UAI', 'Code commune',
                    'Presents - L', 'Presents - ES', 'Presents - S',
                    'Taux de mentions - L', 
                    'Taux de mentions - ES',
                    'Taux de mentions - S']]
        lyc_gen = lyc_gen[(lyc_gen['Presents - L']>0) |
            (lyc_gen['Presents - ES']>0)|
            (lyc_gen['Presents - S']>0)]
        lyc_gen = lyc_gen.fillna(0)
        # Calculate the taux de mention for each lycée
        lyc_gen['taux_mention'] = (lyc_gen['Presents - L'] * lyc_gen['Taux de mentions - L'] + lyc_gen['Presents - ES'] * lyc_gen['Taux de mentions - ES'] + lyc_gen['Presents - S'] * lyc_gen['Taux de mentions - S']) / (lyc_gen['Presents - S'] + lyc_gen['Presents - L'] + lyc_gen['Presents - ES'])
        # Merge the lycée data with the geographical data
        lyc_gen = lyc_gen.merge(geo_etab, how = 'left', left_on = 'UAI', right_on = 'numero_uai')
        # Select only the relevant columns and rename them for clarity
        lyc_gen = lyc_gen[['Etablissement', 'UAI', 'Code commune', 'code_departement',
                'Taux de mentions - L', 'Taux de mentions - ES', 'Taux de mentions - S', 'taux_mention',
                'latitude', 'longitude']]
        lyc_gen.rename(columns = {'Taux de mentions - L':'taux_mention_L', 'Taux de mentions - ES':'taux_mention_ES', 'Taux de mentions - S':'taux_mention_S'}, inplace=True)
        # Convert the resulting DataFrame to a geopandas GeoDataFrame, and filter out any rows with missing geographic data
        lyc_gen_geo = gpd.GeoDataFrame(
            lyc_gen, geometry = gpd.points_from_xy(lyc_gen.longitude, lyc_gen.latitude))
        lyc_gen_geo = lyc_gen_geo[(lyc_gen_geo['latitude'].notna()) & (lyc_gen_geo['longitude'].notna())]

        return lyc_gen_geo
    except Exception as e:
        print(f"An error occurred while preprocessing lycees data: {e}")
        return None


def prep_brevet(data, geo_etab):
    """
    Preprocesses brevet data by computing the taux de mention for each college,
    converting it to a geopandas dataframe, and merging it with the DVF dataframe.
    """
    try:
        brevet = data[data['session'] == 2021]
        brevet_geo = brevet.merge(geo_etab, how = 'left', left_on = 'numero_d_etablissement', right_on = 'numero_uai')
        brevet_geo = brevet_geo[['numero_uai', 'code_commune',
                                'nombre_total_d_admis', 'nombre_d_admis_mention_tb','taux_de_reussite',
                                'latitude', 'longitude']]
        brevet_geo['taux_mention'] = brevet_geo['nombre_d_admis_mention_tb'] / brevet_geo['nombre_total_d_admis']

        brevet_geo = gpd.GeoDataFrame(
            brevet_geo, geometry = gpd.points_from_xy(brevet_geo.longitude, brevet_geo.latitude))
        brevet_geo = brevet_geo[(brevet_geo['latitude'].notna()) & (brevet_geo['longitude'].notna())]

        return brevet_geo

    except TypeError as e:
        print(f"TypeError occurred while preprocessing brevet data: {e}")
        return None
    
    except KeyError as e:
        print(f"KeyError occurred while preprocessing brevet data: {e}")
        return None

