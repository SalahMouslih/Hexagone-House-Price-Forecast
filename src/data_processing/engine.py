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

    # Filtering price
    dvf = fonction_final_prix(dvf, trimestre_actu=trimestre_actu)

    # Train test split 
    dvf_train = dvf.loc[~dvf['trimestre_vente'].isin(test_trimestre)]
    dvf_test = dvf.loc[dvf['trimestre_vente'].isin(test_trimestre)]

    dvf_train = filtre_prix(dvf_train,'prix_m2_actualise')
    #dvf_test = filtre_prix(dvf_test,'prix_m2')
    
    # Concatenate
    #dvf = pd.concat([dvf_train, dvf_test])

    # Convert to geopandas
    dvf_geo = convert_gpd(dvf_train)

    # Create the variable "prix moyen au m2 des 10 biens les plus proches"
    dvf_geo = calculate_closest_metric(dvf = dvf_geo, table_info = dvf_geo[~dvf_geo['trimestre_vente'].isin(test_trimestre)],
            k_neighbors = 10,
            metric_of_interest = 'prix_m2_actualise',
            new_metric_name = 'prix_m2_zone')

    dvf_geo = dvf_geo.reset_index(drop=True)

    ##..
    #lyc_gen_geo = prep_lyc(lyc, geo_etab)
    #dvf_geo = calculate_closest_metric(dvf = dvf_geo,
     #          table_info = lyc_gen_geo,
     #          k_neighbors = 3,
     #          metric_interest = 'taux_mention',
     #          name_new_metric = 'moyenne')

    ## Ecoles
    # Get the taux de mention for each collège + geographical coordinates of schools
    #brevet_geo = prep_brevet(brevet, geo_etab)

    # We get for each good the average taux de mention of the 3 closest collèges
    #dvf_geo = calculate_closest_metric(dvf = dvf_geo,
    #                table_info = brevet_geo,
    #                k_neighbors = 3,
    #                metric_interest = 'taux_mention',
     #               name_new_metric = 'moyenne_brevet')

    # Save the processed data
    output_dir = "data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "processed_data.csv")
    pd.DataFrame(dvf_geo).to_csv(output_file, index=False)
    
    print('Finished pre-processing')
    print('************************')
    print('Processed data saved to', output_dir)

    return True