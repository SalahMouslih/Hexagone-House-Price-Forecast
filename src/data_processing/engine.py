from data_processing.amenities import equipements_prep, read_equi
from data_processing.clean import clean_multivente
from data_processing.discount import fonction_final_prix
from data_processing.education import prep_brevet, prep_lyc
from data_processing.filters import select_bien, filtre_dur, filtre_prix
from data_processing.utilities import calculate_closest_metric, choose_metric_name, iris_prep, get_top_zones,convert_gpd, read_dvfs, read_iris, read_lyc, select_variables
import pandas as pd
import os

trimestre_actu = '2022-T2'
test_trimestre=['2021-T3','2021-T4','2022-T1','2022-T2']
surface_max_maison = 360
surface_max_appartement = 200
nombre_pieces_max_maison = 10
nombre_max_appartement = 6

def preprocessing_engine(data_paths, trimestre_actu=trimestre_actu, test_trimestre=test_trimestre):
    """Main engine of preprocessing. Preprocesses DVF data in an end-to-end fashion."""
    try:
        # Read data
        data = read_dvfs(data_paths)
    except FileNotFoundError:
        print("Error: data file not found")
        return None

    # Select the top 10 metropoles
    data_top = get_top_zones(data,10)
    #Clean the data to keep only multiventes
    clean_data = clean_multivente(data_top)
    
    #Apply filters to select properties of interest
    dvf = select_bien(clean_data)
    dvf = filtre_dur(dvf, surface_max_maison, nombre_pieces_max_maison, 'Maison')
    dvf = filtre_dur(dvf, surface_max_appartement, nombre_max_appartement, 'Appartement')
    # Discounting price
    dvf = fonction_final_prix(dvf, trimestre_actu=trimestre_actu)

    # TSplit the data into training and testing datasets
    dvf_train = dvf.loc[~dvf['trimestre_vente'].isin(test_trimestre)]
    dvf_test = dvf.loc[dvf['trimestre_vente'].isin(test_trimestre)]
  

    # Filter the prices of the datasets
    dvf_train = filtre_prix(dvf_train,'prix_m2_actualise')
    dvf_test = filtre_prix(dvf_test,'prix_m2')
    
    # Concatenate train and test data
    dvf = pd.concat([dvf_train, dvf_test])

    # Convert data to geopandas
    dvf_geo = convert_gpd(dvf_train)

    # Create the variable "prix moyen au m2 des 10 biens les plus proches"
    dvf_geo = calculate_closest_metric(dvf = dvf_geo, table_info = dvf_geo[~dvf_geo['trimestre_vente'].isin(test_trimestre)],
            k_neighbors = 10,
            metric_of_interest = 'prix_m2_actualise',
            new_metric_name = 'prix_m2_zone')
    dvf_geo = dvf_geo.reset_index(drop=True)


    # Get the taux de mention for each lycée and collège as well as their geographical coordinates
    try:
        geo_etab, brevet, lyc = read_lyc()
        lyc_gen_geo = prep_lyc(lyc, geo_etab)
        brevet_geo = prep_brevet(brevet, geo_etab)

        # Calculate the average 'taux de mention' of the 3 closest 'lycées' for each property
        dvf_geo = calculate_closest_metric(dvf=dvf_geo, table_info=lyc_gen_geo,
                                            k_neighbors=3,
                                            metric_of_interest='taux_mention',
                                            new_metric_name='moyenne')

        # Calculate the average 'taux de mention' of the 3 closest 'collèges' for each property
        dvf_geo = calculate_closest_metric(dvf=dvf_geo, table_info=brevet_geo,
                                            k_neighbors=3,
                                            metric_of_interest='taux_mention',
                                            new_metric_name='moyenne_brevet')

    except Exception as e:
        print("Error while calculating 'taux de mention' for lycées and collèges:", str(e))
        return None


    # Add information about the IRIS area
    iris_value, iris_shape = read_iris()
    iris = iris_prep(iris_value, iris_shape)
    dvf_geo = dvf_geo.sjoin(iris, how = 'left', predicate = 'within')

    #Choose the metric name for income
    dvf_geo = choose_metric_name(dvf_geo,'income')


    #Add information about the equipment available in the area
    liste_iris = dvf_geo['DCOMIRIS'].unique()
    equipements = equipements_prep(liste_iris)

    dvf_geo = dvf_geo.merge(equipements, how = 'left', left_on = 'DCOMIRIS', right_on = 'DCIRIS')
    dvf_geo = choose_metric_name(dvf_geo,'amenity')

    # Select the relevant variables
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
            print('****************************')
            print('Processed data saved to', output_dir)

        except IOError:
            print("Error: could not write processed data to file")
            return None

    return True