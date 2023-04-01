import pandas as pd

def clean_type(clean_data, type_bien):
    print(f"Cleaning data for '{type_bien}'")

    mask = clean_data['type_local'] == type_bien
    clean_data = clean_data[mask]
    df = clean_data.groupby('index_group')['numero_disposition'].nunique()
    df = clean_data[clean_data.index_group.isin(df[df==1].index)].groupby('index_group').size()
    to_drop = df[df>1].index
    clean_data = clean_data.drop(clean_data[clean_data.index_group.isin(to_drop)].index)
    
    return clean_data

def clean_multivente(data):
    '''
    Drop duplicates and filter for 'Sale' transactions. Then, remove mutations with multiple disposition IDs, as they are more complex. 
    If there are multiple rows with the same mutation ID, filter by property type (Appartement ou Maison) and keep only one row for each type. 

    '''

    print("Cleaning multivente data...")

    data = data.drop_duplicates()
    data = data[data['nature_mutation'] == 'Vente']
    data['index_group'] = data['id_mutation'].astype(str) + data['date_mutation'].astype(str)
    df = data.groupby('index_group')['numero_disposition'].nunique()
    to_drop = df[df>1].index
    data = data.drop(data[data.index_group.isin(to_drop)].index)
    
    return pd.concat([clean_type(data, 'Appartement'), clean_type(data, 'Maison')])
