from data_processing.amenities import equipements_prep
from data_processing.clean import clean_multivente
from data_processing.discount import fonction_final_prix
from data_processing.education import prep_brevet, prep_lyc
from data_processing.filters import select_bien, filtre_dur, filtre_prix
from data_processing.utilities import calculate_closest_metric, chose_metric_names, get_top_zones,convert_gpd, read_dvfs, read_tables


def preprocessing_engine(data_paths, trimestre_actu='2022-T2', test_trimestre=['2021-T3','2021-T4','2022-T1','2022-T2']):
    """Main engine of preprocessing. Preprocesses DVF data in an end-to-end fashion."""
    try:
        # Read tables
        data = read_dvfs()
        geo_etab, brevet, lyc = read_tables()
    except FileNotFoundError:
        print("Error: data file not found")
        return None

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


    # Get the taux de mention for each lycée and collège as well as their geographical coordinates
    lyc_gen_geo = prep_lyc(lyc, geo_etab)
    brevet_geo = prep_brevet(brevet, geo_etab)

    # Get for each property the average 'taux de mention' of the 3 closest 'lycées'
    dvf_geo = calculate_closest_metric(dvf = dvf_geo, table_info = lyc_gen_geo,
              k_neighbors = 3,
              metric_of_interest = 'taux_mention',
              new_metric_name = 'moyenne')

    # Get for each property the average 'taux de mention' of the 3 closest 'collèges'
    dvf_geo = calculate_closest_metric(dvf = dvf_geo, table_info = brevet_geo,
                    k_neighbors = 3,
                    metric_of_interest = 'taux_mention',
                    new_metric_name = 'moyenne_brevet')

    ##Iris
    iris = iris_prep()
    dvf_geo = dvf_geo.sjoin(iris, how = 'left', predicate = 'within')

    ##
    dvf_geo = chose_metric_name(dvf_geo,'income')


    ##equippement

    bpe = read_equi()
    equipements = equipements_prep(bpe)

    dvf_geo = dvf_geo.merge(equipements, how = 'left', left_on = 'DCOMIRIS', right_on = 'DCIRIS')
    dvf_geo = chose_metric_name(dvf_geo,'equi')

    #
    dvf_geo = select_variables(dvf_geo)

    if dvf_geo is None:
        print("Error: data preprocessing failed")
    else:
        try:
            # Save the processed data
            output_dir = "data/processed"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, "processed_data.csv")
            pd.DataFrame(dvf_geo).to_csv(output_file, index=False)
            print('Finished pre-processing')
            print('************************')
            print('Processed data saved to', output_dir)

        except IOError:
            print("Error: could not write processed data to file")
            return None

    return True