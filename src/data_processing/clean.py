import pandas as pd


def clean_type(data, type_bien):
    """
    Cleans data by removing all properties of a given type (Appartement, Maison, etc.) 
    where the same property has been counted multiple times.
    """
    print(f"Cleaning data for '{type_bien}...'")

    # Filter data by property type
    mask = data['type_local'] == type_bien
    clean_data = data[mask]

    # Find the number of unique disposition numbers for each property group
    df = clean_data.groupby('index_group')['numero_disposition'].nunique()

    # remove any groups where more than one disposition number is found
    df = clean_data[clean_data.index_group.isin(df[df==1].index)].groupby('index_group').size()
    to_drop = df[df>1].index

    # drop any rows associated with the remaining groups
    clean_data = clean_data.drop(clean_data[clean_data.index_group.isin(to_drop)].index)

    return clean_data


def clean_multivente(data):
    '''
    Clean the given dataset by performing the following operations:
    1. Drop duplicates
    2. Filter for 'Vente' transactions
    3. Remove mutations with multiple disposition IDs, as they are more complex. 
    4. If there are multiple rows with the same mutation ID, filter by property type (Appartement ou Maison) and keep only one row for each type.
    '''
    # Print message to indicate that the function has started
    print("Cleaning multivente data...")

    # Drop duplicates and filter for 'vente' transactions
    data = data.drop_duplicates()
    data = data[data['nature_mutation'] == 'Vente']

    # Create a unique identifier for each mutation using ID and date
    data['index_group'] = data['id_mutation'].astype(str) + data['date_mutation'].astype(str)

    # Remove mutations with multiple disposition IDs
    df = data.groupby('index_group')['numero_disposition'].nunique()
    to_drop = df[df>1].index
    data = data.drop(data[data.index_group.isin(to_drop)].index)

    # Filter by property type and keep only one row for each type if there are multiple rows with the same mutation ID
    clean_data = pd.concat([clean_type(data, 'Appartement'), clean_type(data, 'Maison')])

    return clean_data