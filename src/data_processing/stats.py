import pandas as pd
import matplotlib.pyplot as plt


def stat_sur_filtre(data):
    """
    Compute and print the shape of the input dataframe, group the data by metropole and print the resulting sizes,
    then group the data by 'LIBEPCI' and 'type_local' columns, compute the size of each group, and return the resulting dataframe.
    """
    print(f"Data shape: {data.shape}")
    print("Group sizes by LIBEPCI:")
    print(data.groupby(['LIBEPCI']).size())
    return data.groupby(['LIBEPCI', 'type_local']).size()


def stat_before_after(data,clean_data):
    """
    Given two dataframes, the function selects all rows in the first dataframe with 'nature_mutation' equal to "Vente"
    and 'type_local' equal to "Appartement" or "Maison", then groups the resulting data by 'LIBEPCI' and 'type_local'
    columns, computes the size of each group, and plots a horizontal bar chart for each type of property, sorted by decreasing
    percentage difference between the two dataframes. The function also returns a dataframe containing the same information
    in tabular form.
    
    Args:
        data (pandas.DataFrame): The input dataframe containing all sales data.
        clean_data (pandas.DataFrame): The cleaned version of the input dataframe, with potential duplicates and outliers removed.
    """
    # Remove duplicates from the original data
    data.drop_duplicates(inplace=True)
    
    # Filter the data to keep only sales of apartments or houses
    mask = (data.nature_mutation == "Vente") & ((data.type_local == "Appartement") | (data.type_local == "Maison"))
    sales_before_clean = data[mask]
    
    # Compute the total number of sales by EPCI and 'type_local' before and after cleaning
    total_sales_before = stat_sur_filtre(sales_before_clean)
    total_sales_after = stat_sur_filtre(clean_data)
    
    # Compute the percentage change in sales between the two dataframes
    percentage_change = (total_sales_before - total_sales_after) * 100 / total_sales_before
    percentage_change = percentage_change.sort_values(ascending=False)
    
    # Create a dataframe with the percentage change in sales by LIBEPCI and property type
    result = pd.DataFrame(percentage_change).reset_index().set_index('LIBEPCI')
    result.columns = ['type_local', 'pourcentage (%)']
    
    # Plot a horizontal bar chart for each type of property
    for col in ["Appartement", "Maison"]:
        result[result.type_local==col].sort_values(by='pourcentage (%)', ascending=False).plot.barh()
        plt.title(col)
