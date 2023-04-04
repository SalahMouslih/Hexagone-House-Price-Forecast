import pandas as pd
import warnings
warnings.filterwarnings('ignore')


bpe_data_path = "data/open_data/bpe21_ensemble_xy.csv"
liste_equipements = [['A203'],['A206'],['B101','B102','B103','B201','B202','B203','B204','B205','B206'],['C101','C102','C104','C105'],
                     ['C201','C301','C302','C303','C304','C305'],['D201'],['E107','E108','E109'],['F303'],['F307'],['F313']]


def read_equi():
    """
    Reads the amenities table from the open data directory and returns it as a DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing the amenities data.
    
    Raises:
        IOError: If the amenities file cannot be found or read.
    """
    try:
        print("Reading Equipements table...")
        amenities = pd.read_csv(bpe_data_path, delimiter=';')
        return amenities
    except IOError:
        print("Error: could not read amenities file.")
        return None


def equipements_prep(liste_iris):
    """
    Aggregate the number of equipment for selected categories at the IRIS level.

    Parameters:

    liste_iris (list): list of IRIS to include in the aggregation.

    Returns:

    pd.DataFrame: dataframe containing the aggregated number of equipment for the selected categories at the IRIS level.
    
    Raises:
        ValueError: If the input list of IRIS is empty.
    """
    
    print("Adding amenities...")
    
    # Read amenities file
    amenities = read_equi()
    
    # Filter the amenities dataframe to only include IRIS of interest
    amenities = amenities[amenities['DCIRIS'].isin(liste_iris)]
    amenities_df = []

    for equipement in liste_equipements:
        # Filter the amenities dataframe to only include the current equipment category
        amenities = amenities[amenities['TYPEQU'].isin(equipement)]

        # Group the amenities dataframe by DCIRIS and TYPEQU, count the number of occurrences and store the result in a dataframe
        amenities = amenities.groupby('DCIRIS')['TYPEQU'].value_counts().to_frame()

        # Group the amenities dataframe by DCIRIS, sum the number of equipment and rename the column to the first equipment name in the list
        amenities = amenities.groupby('DCIRIS').sum()
        amenities = amenities.rename(columns={"TYPEQU": equipement[0]})

        # Append the amenities dataframe to the amenities list
        amenities_df.append(amenities_df)

    # Concatenate the amenities dataframes in the amenities list, fill the missing values with 0, and reset the ind
    amenities_df = pd.concat(amenities_df).fillna(0)
    amenities_df['DCIRIS'] = amenities_df.index
    amenities_df = amenities_df.reset_index(drop=True)
    
    #
    amenities_df = amenities_df.drop_duplicates()
    amenities_df = amenities_df.groupby(["DCIRIS"], as_index=False).sum()

    return amenities_df