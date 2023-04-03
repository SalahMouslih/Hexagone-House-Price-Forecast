import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#
liste_equipements = [['A203'],['A206'],['B101','B102','B103','B201','B202','B203','B204','B205','B206'],['C101','C102','C104','C105'],
                     ['C201','C301','C302','C303','C304','C305'],['D201'],['E107','E108','E109'],['F303'],['F307'],['F313']]


def equipements_prep(amenities, liste_equipements = liste_equipements):
    """
    Aggregate the number of equipment for selected categories at the IRIS level.

    Parameters:

    liste_equipements (list): list of equipment categories to consider.
    Returns:

    amenities (pandas dataframe): dataframe containing the aggregated number of equipment for the selected categories at the IRIS level.
    """
    
    print("Adding amenities...")
    
    # Filter the amenities dataframe to only include IRIS of interest
    amenities_df = amenities[amenities_df['DCIRIS'].isin(liste_iris)]
    amenities = []

    for equipement in liste_equipements:
        # Filter the amenities dataframe to only include the current equipment category
        amenities_df = amenities_df[amenities_df['TYPEQU'].isin(equipement)]

        # Group the amenities dataframe by DCIRIS and TYPEQU, count the number of occurrences and store the result in a dataframe
        amenities_df = amenities_df.groupby('DCIRIS')['TYPEQU'].value_counts().to_frame()

        # Group the amenities dataframe by DCIRIS, sum the number of equipment and rename the column to the first equipment name in the list
        amenities_df = amenities_df.groupby('DCIRIS').sum()
        amenities_df = amenities_df.rename(columns={"TYPEQU": equipement[0]})

        # Append the amenities dataframe to the amenities list
        amenities.append(amenities_df)

    # Concatenate the amenities dataframes in the amenities list, fill the missing values with 0, and reset the ind
    amenities = pd.concat(amenities).fillna(0)
    amenities['DCIRIS'] = amenities.index
    amenities = amenities.reset_index(drop=True)
    
    #
    amenities = amenities.drop_duplicates()
    amenities = amenities.groupby(["DCIRIS"], as_index=False).sum()

    return amenities