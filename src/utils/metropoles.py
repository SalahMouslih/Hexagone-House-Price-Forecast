import pandas as pd

# Import the table which defines the metrpole (EPCI)
path=#enter path
metropoles = pd.read_csv(path, delimiter=';', header=5)

def get_top_zones(df, nb_top_zones):

    """Select zones where the highest number mutations"""

    # Correct the spelling of regions
    df.loc[df.nom_commune.str.startswith('Marseille '), 'nom_commune'] = 'Marseille'
    df.loc[df.nom_commune.str.startswith('Lyon '), 'nom_commune'] = 'Lyon'
    df.loc[df.nom_commune.str.startswith('Paris '), 'nom_commune'] = 'Paris'

    # Merge dvf and metropole
    df = df.merge(metropoles, how='left', left_on='nom_commune', right_on='LIBGEO')

    # Pick the areas with the highest number of transactions
    most_frequent = df['LIBEPCI'].value_counts().head(nb_top_zones).index.to_list()
    df = df.loc[df['LIBEPCI'].isin(most_frequent)]

    return df
