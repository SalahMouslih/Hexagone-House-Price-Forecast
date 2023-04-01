from data_processing.clean import *
from data_processing.filtres import *
from data_processing.discount import *
from data_processing.utilities import *


def preprocess_dvf_data(data_paths, trimestre_actu='2022-T2', test_trimestre=['2021-T3','2021-T4','2022-T1','2022-T2']):
    """
    Preprocesses DVF data and returns a geopandas dataframe.

    Args:
        data_path (str): Path to input DVF data.
        trimestre_actu (str): The trimester to use for the final price calculation.
        test_trimestre (list): List of trimesters to use for testing data split. Default is ['2021-T3','2021-T4','2022-T1','2022-T2'].

    Returns:
        dvf_geo (geopandas.DataFrame): Preprocessed DVF data as a geopandas dataframe.
    """
    # Import DVFs
    data = read_dvfs(data_paths)

    # Select metropoles
    data_top = get_top_zones(data,10)

    # Keep multiventes
    clean_data = clean_multivente(data_top)
    
    # Apply filters
    dvf = select_bien(clean_data)
    dvf = filtre_dur(dvf, 360, 10, 'Maison')
    dvf = filtre_dur(dvf, 200, 6, 'Appartement')

    #
    func = fonction_final_prix(dvf,trimestre_actu=trimestre_actu,actulisation=False)

    # Observe year by year to choose the split date 
    find_pourcentage=(func['trimestre_vente'].value_counts(normalize=True)).sort_index().cumsum()
    find_pourcentage

    find_pourcentage[find_pourcentage>0.8]

    #
    dvf = fonction_final_prix(dvf,trimestre_actu=trimestre_actu)

    # Train test split 
    dvf_train = dvf[~dvf['trimestre_vente'].isin(test_trimestre)]
    dvf_test = dvf[dvf['trimestre_vente'].isin(test_trimestre)]
    
    dvf_train = filtre_prix(dvf_train,'prix_m2_actualise', 0.99)
    dvf_test = filtre_prix(dvf_test,'prix_m2', 0.99)
    
    # Concatenate
    dvf = pd.concat([dvf_train, dvf_test])

    # Convert to geopandas
    dvf_geo = convert_gpd(dvf)

    # Create the variable "prix moyen au m2 des 10 biens les plus proches"
    dvf_geo = my_choose_closest(dvf = dvf_geo, table_info = dvf_geo[~dvf_geo['trimestre_vente'].isin(test_trimestre)],
            k_neighbors = 10,
            metric_interest = 'prix_m2_actualise',
            name_new_metric = 'prix_m2_zone')

    dvf_geo = dvf_geo.reset_index(drop=True)

    # Save the processed data
    output_dir = "data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "processed_data.csv")
    dvf_geo.to_csv(output_file, index=False)

    return True