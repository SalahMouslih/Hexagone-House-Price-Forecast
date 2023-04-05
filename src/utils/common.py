import pandas as pd
import geopandas as gpd


def read_data(data_paths):
    """
    Read data from the given path(s) and return a single concatenated dataframe.
    """
    try:
        print('Reading data...')
        # check if input is a single file path or a list of file paths
        if isinstance(data_paths, str):
            data = pd.read_csv(data_paths)
        else:
            data = pd.concat(map(pd.read_csv, data_paths))
        return data
    except FileNotFoundError as e:
        print(f"Error occurred while reading data: {e}")
        return None
    except Exception as e:
        print(f"Error occurred while reading data: {e}")
        return None

def read_tables(*data_paths):
    """
    Read multiple csv files from the given paths and return a list of dataframes.
    """
    dataframes = [] 

    # iterate over each path in the input arguments
    for path in data_paths: 
        dataframe = pd.read_csv(path) 
        # add the resulting dataframe to the list
        dataframes.append(dataframe) 

    # return the list of dataframes
    return dataframes 

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

def convert_gpd(df, equi=False):
    """
    Function convert_gpd converts a pandas DataFrame to a GeoDataFrame using the geometry
    attribute which is created from the longitude and latitude columns of the input DataFrame
    """
    try:
        if equi:
            return gpd.GeoDataFrame(
                df, geometry = gpd.points_from_xy(df.LAMBERT_X, df.LAMBERT_Y)
            )
        else :
            return gpd.GeoDataFrame(
                df, geometry = gpd.points_from_xy(df.longitude, df.latitude)
            )
    except ValueError as e:
        print(f"Error converting to GeoDataFrame: {e}")
        return None

def read_iris():
    """
    Read IRIS tables and return iris_value and iris_shape.
    """
    iris_value_path = 'data/open_data/IRIS_donnees.csv'
    iris_shape_path = 'data/open_data/IRIS_contours.shp'
    try:
        print("Reading 'iris' tables...this might take a while")
        iris_value = pd.read_csv(iris_value_path, delimiter=';')
        iris_shape = gpd.read_file(iris_shape_path)
        return iris_value, iris_shape
    except FileNotFoundError as e:
        print(f"Error occurred while reading data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_equi():
    """
    Reads the amenities table from the open data directory and returns it as a DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing the amenities data.
    Raises:
        IOError: If the amenities file cannot be found or read.
    """
    bpe_data_path = "data/open_data/bpe21_ensemble_xy.csv"
    try:
        print("Reading 'equipements' table...")
        # Read 'base permanente des equipements' file
        amenities = pd.read_csv(bpe_data_path, delimiter=';')
        return amenities
    except IOError:
        print("Error: could not read amenities file.")
        return None

def iris_prep(iris_value, iris_shape):
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
