"""
This module contains functions for discounting real estate data.

The main function in this module is 'create_columns', which computes 
the discount'coeff_appart_a_maison' and 'coeff_maison_a_appart' and
apply discounting.
"""
from datetime import datetime
import numpy as np
import pandas as pd
import swifter
from tqdm import tqdm

# Define paths
PATH_VALEURS_TRIMESTROIELLES = "data/open_data/valeurs_trimestrielles.csv"
PATH_ZONAGE_IMMO = "data/open_data/Zonage_abc_communes_2022.xlsx"

# Define lists of indices
liste_grande_ville=['Indice des prix des logements anciens - Agglomération de Marseille - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS',
                   'Indice des prix des logements anciens - Agglomération de Lille - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS',
                   'Indice des prix des logements anciens - Agglomération de Lyon - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS',
                   'Indice des prix des logements anciens - Paris - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS',
                   'Indice des prix des logements anciens - France métropolitaine - Appartements - Base 100 en moyenne annuelle 2015 - série CVS',
                   'Indice des prix des logements anciens - France métropolitaine - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS',
                   "Indice des prix des logements anciens - Zone A du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS",
                   "Indice des prix des logements anciens - Zone A bis du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS",
                   "Indice des prix des logements anciens - Zone B1 du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS",
                   "Indice des prix des logements anciens - Zone B2 du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS",
                   "Indice des prix des logements anciens - Zone C du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS"]

liste_Marseille=['Marseille 2e Arrondissement',
       'Marseille 3e Arrondissement', 'Marseille 1er Arrondissement',
       'Marseille 15e Arrondissement', 'Marseille 14e Arrondissement',
       'Marseille 4e Arrondissement', 'Marseille 16e Arrondissement',
       'Marseille 7e Arrondissement', 'Marseille 10e Arrondissement',
       'Marseille 6e Arrondissement', 'Marseille 5e Arrondissement',
       'Marseille 8e Arrondissement', 'Marseille 9e Arrondissement',
       'Marseille 12e Arrondissement', 'Marseille 13e Arrondissement',
       'Marseille 11e Arrondissement']
                 
liste_lyon=['Lyon 9e Arrondissement',
       'Lyon 1er Arrondissement', 'Lyon 2e Arrondissement',
       'Lyon 5e Arrondissement', 'Lyon 4e Arrondissement',
       'Lyon 8e Arrondissement', 'Lyon 3e Arrondissement',
       'Lyon 7e Arrondissement', 'Lyon 6e Arrondissement']
                 
liste_paris=['Paris 8e Arrondissement',
       'Paris 3e Arrondissement', 'Paris 1er Arrondissement',
       'Paris 18e Arrondissement', 'Paris 7e Arrondissement',
       'Paris 5e Arrondissement', 'Paris 6e Arrondissement',
       'Paris 11e Arrondissement', 'Paris 13e Arrondissement',
       'Paris 10e Arrondissement', 'Paris 9e Arrondissement',
       'Paris 12e Arrondissement', 'Paris 14e Arrondissement',
       'Paris 15e Arrondissement', 'Paris 16e Arrondissement',
       'Paris 17e Arrondissement', 'Paris 20e Arrondissement',
       'Paris 19e Arrondissement', 'Paris 2e Arrondissement',
       'Paris 4e Arrondissement']

liste_complete = liste_paris + liste_lyon + liste_Marseille


def create_columns(data):
    """Create columns 'Appartement' and 'Maison' and assign values based on 
    'coeff_appart_a_maison' and 'coeff_maison_a_appart' columns.
    """
    # Capture columns
    var_list = data.columns

    for i in var_list:
        if 'Appartement' in i:
            new_var = i.replace('Appartement', 'Maison')
            if new_var not in var_list:
                data[new_var] = data[i] * data['coeff_appart_a_maison']
                
        if 'Maison' in i:
            new_var = i.replace('Maison', 'Appartement')
            if new_var not in var_list:
                data[new_var] = data[i] * data['coeff_maison_a_appart']

    return data


def commune(x):
    """Convert the input string into a valid commune code string."""
    commune_name = str(x)

    if len(commune_name) == 4:
        new_commune = '0' + commune_name

    return new_commune


def fill_zone(data: pd.DataFrame):
    """Fill the missing values of 'Zone ABC' with the corresponding city or 'C' 
    if the city is not in the list.
    """
    
    commune_name = data['nom_commune']
    zone = ''

    if commune_name in liste_Marseille:
        zone = 'Marseille'
    elif commune_name in liste_lyon:
        zone = 'Lyon'
    elif commune_name in liste_paris:
        zone = 'Paris'
    elif commune_name == 'Lille':
        zone = 'Lille'
    elif pd.isna(data['Zone ABC']):
        if commune_name not in liste_complete:
            zone = 'C'
    else:
        zone = data['Zone ABC']

    return zone


def get_trimester(data):
    """Get the trimester based on the date"""

    date = data['date_vente']
    month = int(date.month)
    year = date.year
    trimester = ''

    if month < 4:
        trimester = 'T1'
    elif 4 <= month < 7:
        trimester = 'T2'
    elif 7 <= month < 10:
        trimester = 'T3'
    else:
        trimester = 'T4'

    trim_vente = str(year) + '-' + str(trimester)

    return trim_vente

def get_coeff_actu(data, base_indice_grand, trimestre_actu):
    """
    Calculate the discount coefficient for a given zone and type of property.

    Args:
        data: A dictionary containing the zone, trimester, and type of property.
        base_indice_grand: A pandas DataFrame of the base index data for the zone and type of property.
        trimestre_actu: A string representing the trimester of discount.

Returns:
- coeff: A float representing the coefficient of price evolution.
    """
    try:
        # Get the zone, trimester, and type of property from the data
        zone = data['vrai_zone']
        trimestre = data['trimestre_vente']
        type_bien = data['type_local']

        ligne = ''

        # Determine the correct line in the base index based on the zone and type of property
        if zone == 'Paris':
            if type_bien == 'Appartement':
                ligne = 'Indice des prix des logements anciens - Paris - Appartements - \
                        Base 100 en moyenne annuelle 2015 - Série CVS'
            else:
                ligne = 'Indice des prix des logements anciens - Paris - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS'
        elif zone == 'Marseille':
            if type_bien == 'Appartement':
                ligne = 'Indice des prix des logements anciens - Agglomération de Marseille - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS'
            else:
                ligne = 'Indice des prix des logements anciens - Agglomération de Marseille - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS'
        elif zone == 'Lyon':
            if type_bien == 'Appartement':
                ligne = 'Indice des prix des logements anciens - Agglomération de Lyon - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS'
            else:
                ligne = 'Indice des prix des logements anciens - Agglomération de Lyon - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS'
        elif zone == 'Lille':
            if type_bien == 'Appartement':
                ligne = 'Indice des prix des logements anciens - Agglomération de Lille - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS'
            else:
                ligne = 'Indice des prix des logements anciens - Agglomération de Lille - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS'
        elif zone == 'A':
            ligne = "Indice des prix des logements anciens - Zone A du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS"
        elif zone == 'Abis':
            ligne = "Indice des prix des logements anciens - Zone A bis du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS"
        elif zone == 'B1':
            ligne = "Indice des prix des logements anciens - Zone B1 du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS"
        elif zone == 'B2':
            ligne="Indice des prix des logements anciens - Zone B2 du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS"
        elif zone=='C':
            ligne="Indice des prix des logements anciens - Zone C du Zonage A, B, C - Base 100 en moyenne annuelle 2015 - Série CVS"
        else:
            raise ValueError('Invalid zone:', zone)

        # Select the row of interest from the dataframe
        # The row is identified by the 'Libellé' column matching the 'ligne' string
        données = base_indice_grand[base_indice_grand['Libellé'].isin([ligne])]

        # Extract the index values for the current and base quarters
        # Convert the values to floats
        indice_ancien = données[trimestre].apply(lambda x: float(x))
        indice_actu = données[trimestre_actu].apply(lambda x: float(x))

        # The coefficient represents the ratio of the current index to the base index
        # It is calculated as (current index - base index) / base index + 1
        coeff = float(((indice_actu - indice_ancien) / indice_ancien) + 1)

        # Return the coefficient
        return coeff
    except KeyError as e:
        print(f"KeyError occurred: {str(e)}. The data dictionary may not have the expected key.")
        return None

def fonction_final_prix(data, trimestre_actu, actulisation=True):

    """
    Compute the updated real estate price per square meter using the actualisation coefficient.

    Args:
        data (pd.DataFrame): The real estate data to be processed.
        trimestre_actu (str): The discounted quarter.
        actulisation (bool, optional): Whether to apply actualisation or not. Defaults to True.

    Returns:
        pd.DataFrame: The joined data with the updated real estate price per square meter.
    """
    try:
        # Process the real estate indices table
        base_indice = pd.read_csv(PATH_VALEURS_TRIMESTROIELLES,sep=';')
        base_indice = base_indice[['Libellé','2016-T1', '2016-T2', '2016-T3', '2016-T4', '2017-T1', '2017-T2',
        '2017-T3', '2017-T4', '2018-T1', '2018-T2', '2018-T3', '2018-T4',
        '2019-T1', '2019-T2', '2019-T3', '2019-T4', '2020-T1', '2020-T2',
        '2020-T3', '2020-T4', '2021-T1', '2021-T2', '2021-T3', '2021-T4',
        '2022-T1', '2022-T2', '2022-T3']]

        base_indice_grand=base_indice[base_indice['Libellé'].isin(liste_grande_ville)]
        base_indice_grand.set_index('Libellé',inplace=True)
        base_indice_grand=base_indice_grand.transpose()

        base_indice_grand = base_indice_grand.replace('(s)', np.nan)
        base_indice_grand = base_indice_grand.fillna( method='ffill',)
        base_indice_grand = base_indice_grand.astype('float')

        # Create the coefficient variables
        base_indice_grand['coeff_maison_a_appart'] = base_indice_grand['Indice des prix des logements anciens - France métropolitaine - Appartements - Base 100 en moyenne annuelle 2015 - série CVS']/base_indice_grand['Indice des prix des logements anciens - France métropolitaine - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS']
        base_indice_grand['coeff_appart_a_maison'] = base_indice_grand['Indice des prix des logements anciens - France métropolitaine - Maisons - Base 100 en moyenne annuelle 2015 - Série CVS']/base_indice_grand['Indice des prix des logements anciens - France métropolitaine - Appartements - Base 100 en moyenne annuelle 2015 - série CVS']

        base_indice_grand=create_columns(base_indice_grand)

        liste_drop=['coeff_maison_a_appart', 'coeff_appart_a_maison','Indice des prix des logements anciens - France métropolitaine - Maisons - Base 100 en moyenne annuelle 2015 - série CVS',
        'Indice des prix des logements anciens - France métropolitaine - Appartements - Base 100 en moyenne annuelle 2015 - Série CVS']
        base_indice_grand = base_indice_grand.drop(columns=liste_drop)
        base_indice_grand=base_indice_grand.transpose()
        base_indice_grand = base_indice_grand.reset_index()

        # Import of the real estate areas table
        zone = pd.read_excel(PATH_ZONAGE_IMMO, engine='openpyxl')
        zone = zone.rename(columns={'Nom Commune': 'nom_commune'})

        # Join dvf and area table, then replace missing values
        data['Code Commune'] = data['code_commune'].apply(lambda x: commune(x)).astype("str")
        joined_data = pd.merge(data, zone,how="left", on='Code Commune')
        
        joined_data['vrai_zone'] = joined_data.apply(lambda x: fill_zone(x),axis=1)
        
        # Get the sale trimesters
        joined_data['date_vente'] = joined_data['date_mutation'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        joined_data['trimestre_vente'] = joined_data.apply(lambda x: get_trimester(x),axis=1)
        
        if actulisation:
        
            print('Starting discount...')
        
            # Compute the actualisation coefficient
            joined_data['coeff_actu'] = list(tqdm(joined_data.swifter.apply(lambda x: get_coeff_actu(x,base_indice_grand,trimestre_actu),axis=1),
                                        total=len(joined_data)))
            drop_zone_list = ['Zone ABC','vrai_zone','date_vente']
            joined_data = joined_data.drop(columns=drop_zone_list)

            # Add columns
            
            # Create the target variables
            joined_data['prix_actualise'] = joined_data['valeur_fonciere'] * joined_data['coeff_actu']
            joined_data['prix_m2_actualise'] = joined_data['prix_actualise'] / joined_data['surface_reelle_bati']
            joined_data['prix_m2'] = joined_data['valeur_fonciere'] / joined_data['surface_reelle_bati']
            
        return joined_data        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
