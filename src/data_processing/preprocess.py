from clean import *
from filtres import *
from discount import *
from utilities import *


test_trimestre=['2021-T3','2021-T4','2022-T1','2022-T2']

def preprocess(filename):
    
    # Import DVFs
    data=pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "/Challenge/GROUP4/TEDONZE/base_de_données/datadvf20*.csv"))))

    # Select metropoles
    data_top = zone_top(data,10)

    # Keep multiventes
    clean_data = clean_multivente(data_top)
    
    # Apply filters
    dvf = select_bien(clean_data)
    dvf = filtre_dur(dvf, 360, 10, 'Maison')
    dvf = filtre_dur(dvf, 200, 6, 'Appartement')

    #
    func = fonction_final_prix(dvf,trimestre_actu='2022-T2',actulisation=False)

    # Observe year by year to choose the split date 
    find_pourcentage=(func['trimestre_vente'].value_counts(normalize=True)).sort_index().cumsum()
    find_pourcentage

    find_pourcentage[find_pourcentage>0.8]

    #
    dvf = fonction_final_prix(dvf,trimestre_actu='2021-T2')

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

    return data
