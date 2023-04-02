import pandas as pd

to_drop=['adresse_numero', 'adresse_suffixe','numero_disposition',
       'adresse_code_voie', 'code_postal', 'code_commune',
       'ancien_code_commune','ancien_nom_commune' , 'ancien_id_parcelle',
       'numero_volume', 'lot1_numero', 'lot1_surface_carrez', 'lot2_numero',
       'lot2_surface_carrez', 'lot3_numero', 'lot3_surface_carrez',
       'lot4_numero', 'lot4_surface_carrez', 'lot5_numero',
       'lot5_surface_carrez','code_type_local',
       'code_nature_culture', 'nature_culture', 'code_nature_culture_speciale',
       'nature_culture_speciale','id_mutation','id_parcelle','numero_disposition', 'nature_mutation','valeur_fonciere',
       'id_parcelle','nature_mutation','date_mutation','LIBEPCI','prix_actualise','DCOMIRIS','DCIRIS','prix_m2',
       'type_local','geometry','indices','quantile_prix','coeff_actu']


def pre_process(data, type_local):
    
    # Drop unnecessary columns
    drop_clean=list(set(data.columns)&set(to_drop))
    data  = data.drop(drop_clean, axis=1)
    
    # get numerical columns and calculate correlation matrix
    numerical_columns = list(data.select_dtypes(exclude=["object","string"]).columns)
    target='prix_m2_actualise'
    numerical_columns=[col for col in numerical_columns if col!=target]
    corr_matrix = data[numerical_columns].corr().abs()

    # Remove highly correlated columns
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    data = data.drop(to_drop, axis=1)

    # Drop surface_terrain column for apartments
    if type_local=="Appartement":
        data = data.drop(['surface_terrain'], axis=1)
        
    # Drop rows with missing values
    data = data.dropna()

    return data
