import pandas as pd
import geopandas as gpd


def convert_gpd(df):
    """
    Function convert_gpd converts a pandas DataFrame to a GeoDataFrame using the geometry
    attribute which is created from the longitude and latitude columns of the input DataFrame
    """
    try:
        return gpd.GeoDataFrame(
            df, geometry = gpd.points_from_xy(df.longitude, df.latitude)
        )
    except ValueError as e:
        print(f"Error converting to GeoDataFrame: {e}")
        return None


def iris_prep(iris_value, iris_shape, value_on, shape_on):

  iris_shape.drop_duplicates(subset=['DCOMIRIS'], keep = 'first', inplace = True)
  iris_value.drop_duplicates(subset=['IRIS'], keep = 'first', inplace = True)
  iris_value[value_on] = iris_value[value_on].astype(str).str.rjust(9, '0')

  # merge iris_shape et iris_value pour avoir les polygones dans la même table que les variables de chaque IRIS
  iris = iris_shape.merge(iris_value, how = 'left', right_on = value_on, left_on = shape_on)
  iris.drop_duplicates(subset=['DCOMIRIS'], keep = 'first', inplace = True)

  return iris

