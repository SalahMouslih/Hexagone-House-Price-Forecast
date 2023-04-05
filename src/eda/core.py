"""
EDA Core functions
This module contains core functions for visualizing and analyzing processed data.
"""
import os
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from tqdm import tqdm

def plot_heatmap(data, output_dir=None):
  """
  Plot a heatmap of the correlation matrix between features and "prix_m2_actualise" for a given dataset, 
  either for all metropoles or for a specific metropole, and save the resulting plot to a directory.

  Args:
    data (pandas.DataFrame): Input data containing the columns to be used for computing the correlation matrix.
    output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. 
    If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
    None
  """

  try:
    print("Computing correlation heatmaps per metropole...")

    color = sns.diverging_palette(29, 29, center='light', sep=1, n=40)
    plt.figure(figsize=(8, 12))

    # Plot heatmap for a specific metropole
    for metropole in data.LIBEPCI.unique():
        heatmap = sns.heatmap(data[data.LIBEPCI == metropole].corr()[['prix_m2_actualise']].sort_values(by='prix_m2_actualise', ascending=False), 
                              vmin=-1, vmax=1, annot=True, cmap=color)
        heatmap.set_title("Features Correlating with {}, {}".format('prix_m2_actualise', metropole), fontdict={'fontsize': 18}, pad=16)
        if output_dir:
          filename = os.path.join(output_dir + f"correlation_heatmap_{metropole}.png")
          plt.savefig(filename, dpi=300, bbox_inches="tight")
        else:
          plt.show()
  except Exception as e:
      print(f"Error occurred while plotting heatmap: {e}")

def plot_correlation_matrix(data, output_dir=None):
  """
  Compute the correlation matrix and plot it as a heatmap.

  Args:
    data (pandas.DataFrame): Input data.
    output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. 
    If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
    None
  """

  print("Computing the correlation matrix...")

  # Compute the correlation matrix
  corr = data.corr()

  # Generate a mask for the upper triangle
  mask = np.triu(np.ones_like(corr, dtype=bool))

  # Set up the matplotlib figure and title
  fig, ax = plt.subplots(figsize=(11, 9))
  ax.set(title='Correlation Matrix')

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(230, 100, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
  
  if output_dir:
    # Save the resulting plot to the specified directory
    filename = os.path.join(output_dir, "correlation_matrix.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
  else:
    plt.show()


def distribution_target_type_and_metropoles(data, output_dir=None):
  """
    Plot the distribution of prix_m2_actualise for different types of properties (flats and houses) for all metropoles, 
    and save the resulting plot to a directory.

  Args:
    data (pandas.DataFrame): Input data containing the column "prix_m2_actualise" and "type_local".
    output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. 
    If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
  None
  """
  try:

      print("Generating target variable distribution per metropole...")
      # Iterate over the unique values of "data.LIBEPCI" and plot the distribution of "prix_m2_actualise"
      # for each metropole on a separate subplot
      for _, met in enumerate(data.LIBEPCI.unique()):
          fig = sns.displot(data=data[data.LIBEPCI == met], x="prix_m2_actualise", hue="type_local", kde=True)
          fig.set(title='Distribution of prix_m2_actualise for flats and houses, {}'.format(met))
          if output_dir:
            # Save the resulting plot to the specified directory
            filename = os.path.join(output_dir, f"prix_m2_distribution_{met}.png")
            fig.savefig(filename, dpi=300, bbox_inches="tight")
          else:
            plt.show()
  except Exception as e:
      print(f"Error occurred while creating the plots: {e}")


def plot_flats_houses_shares(data, output_dir=None):
  """
  Plot the percentage of flats and houses for each metropole, and save the resulting plot to a directory.

  Args:
    data (pandas.DataFrame): Input data containing the columns "LIBEPCI" and "type_local".
    output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. 
    If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
    None
  """

  print("Generating percentage of flats and houses for each metropole...")

  # Set the figure size
  plt.figure(figsize=(10, 5))

  # Calculate the percentage of flats and houses for each LIBEPCI
  total = data.groupby('LIBEPCI')['type_local'].count().reset_index()
  type_local = data[data.type_local=='Appartement'].groupby('LIBEPCI')['type_local'].count().reset_index()
  type_local['type_local'] = [i / j * 100 for i,j in zip(type_local['type_local'], total['type_local'])]
  total['type_local'] = [i / j * 100 for i,j in zip(total['type_local'], total['type_local'])]

  # Create a bar chart for flats and houses
  bar1 = sns.barplot(x="LIBEPCI", y="type_local", data=total, color='orange')
  bar2 = sns.barplot(x="LIBEPCI", y="type_local", data=type_local, color='grey')

  # Add legend
  top_bar = mpatches.Patch(color='orange', label='Maison')
  bottom_bar = mpatches.Patch(color='grey', label='Appartement')
  plt.legend(handles=[top_bar, bottom_bar])

  # Rotate x-axis labels for better readability
  plt.xticks(rotation=40)

  # Set plot title
  plt.title("Percentage of Flats and Houses by Metropole")

  if output_dir:
    # Save the resulting plot to the specified directory
    filename = os.path.join(output_dir, "flats_houses_shares.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
  else:
    plt.show()


def box_flats_houses(data, output_dir=None):
  """
  Create and display box plots for the features 'surface_reelle_bati' and 'nombre_pieces_principales'
  for different types of properties (flats and houses) using the given data.

  Args:
    data (pandas.DataFrame): Input data containing the columns 'type_local', 'surface_reelle_bati', and 'nombre_pieces_principales'.
    output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
    None
  """

  try:
    print("Generating box plots of 'surface_reelle_bati' and 'nombre_pieces_principales'...")

    fig = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, x="type_local", y="surface_reelle_bati", palette=['grey', 'orange'])
    plt.xlabel('Type of Property')
    plt.ylabel('Surface Reelle Bati')
    plt.title('Boxplot of Surface Reelle Bati for Flats and Houses')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=data, x="type_local", y="nombre_pieces_principales", palette=['grey', 'orange'])
    plt.xlabel('Type of Property')
    plt.ylabel('Nombre Pieces Principales')
    plt.title('Boxplot of Nombre Pieces Principales for Flats and Houses')

    if output_dir:
      filename = os.path.join(output_dir, "boxplots.png")
      fig.savefig(filename, dpi=300, bbox_inches="tight")
    else:
      plt.show()
  except Exception as e:
    print(f"Error occurred while creating and saving box plots: {e}")


def boxen_flats_houses(data, output_dir=None):
  """
  Generate boxen plots for the surface_reelle_bati and nombre_pieces_principales columns, comparing flats and houses.

  Args:
    data (pandas.DataFrame): Input data containing the columns "type_local", "surface_reelle_bati", and "nombre_pieces_principales".
    output_dir (str, optional): Path to the directory where the figure will be saved. If None, the figure will be displayed using plt.show(). Default is None.

  Returns:
    None
  """

  try:
    print("Generating boxen plots of 'surface_reelle_bati' and 'nombre_pieces_principales'...")

    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    # Generate the boxen plot for surface_reelle_bati column, comparing flats and houses
    sns.boxenplot(data=data, x="type_local", y="surface_reelle_bati", palette=['grey', 'orange'])
    plt.xlabel("Type of Property")
    plt.ylabel("Surface Reelle Bati")
    plt.title("Boxen Plot of Surface Reelle Bati for Flats and Houses")

    plt.subplot(1, 2, 2)

    # Generate the boxen plot for nombre_pieces_principales column, comparing flats and houses
    sns.boxenplot(data=data, x="type_local", y="nombre_pieces_principales", palette=['grey', 'orange'])
    plt.xlabel("Type of Property")
    plt.ylabel("Nombre Pieces Principales")
    plt.title("Boxen Plot of Nombre Pieces Principales for Flats and Houses")

    # Save the figure to the specified directory or display using plt.show()
    if output_dir:
      filename = os.path.join(output_dir, "boxen_plot.png")
      plt.savefig(filename, dpi=300, bbox_inches="tight")
    else:
      plt.show()

  except Exception as e:
    print(f"Error occurred while generating boxen plots: {e}")

def box_flats_houses_metropoles(data, output_dir=None):
  """
  Plot boxenplots of surface_reelle_bati and nombre_pieces_principales for different types of properties (flats and houses) 
  and metropoles, and save the resulting plot to a directory.

  Args:
    data (pandas.DataFrame): Input data containing the columns "type_local", "surface_reelle_bati", "nombre_pieces_principales", and "LIBEPCI".
    output_dir (str): Path to the directory where the plot will be saved.

  Returns:
    None
  """
  try:
    print("Generating boxen plots of 'surface_reelle_bati' and 'nombre_pieces_principales' per metropole...")
    # Define color palette
    palette = sns.dark_palette("orange", 10)
  
    # Set up the figure
    fig = plt.figure(figsize=(10,15))

    # Plot boxenplot for surface_reelle_bati
    plt.subplot(2, 1, 1)
    ax = sns.boxenplot(data=data, x="type_local", y="surface_reelle_bati", hue="LIBEPCI", palette=palette)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Plot boxenplot for nombre_pieces_principales
    plt.subplot(2, 1, 2)
    ax2 = sns.boxenplot(data=data, x="type_local", y="nombre_pieces_principales", hue="LIBEPCI", palette=palette)
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))

    if output_dir:
      # Save the resulting plot to the specified directory
      filename = os.path.join(output_dir, "boxenplot_par_metropole.png")
      fig.savefig(filename, dpi=300, bbox_inches="tight")
    else:
      plt.show()
  except Exception as e:
    print("Error occurred while plotting boxenplots: ", e)
  
def plot_var_iris(iris, area, var, output_dir=None):

  """
  Plot a variable for a specific area using the IRIS dataset.

  Args:
    iris (geopandas.GeoDataFrame): The GeoDataFrame containing IRIS information.
    area (str): The name of the area to plot.
    var (str): The name of the variable to plot.
    output_dir (str, optional): The output directory to save the plot to.

  Returns:
    None

  Raises:
    ValueError: If the specified area is not found in the IRIS dataset.
    Exception: If an error occurs while plotting the variable.
  """
  try:
    print(f"Generating 'prix m2 moyen' for {area}...")

    cmap = plt.get_cmap("jet")
    data = iris[iris.NOM_COM == area]
    if data.empty:
      raise ValueError(f"The specified area '{area}' was not found in the IRIS dataset.")
    figure = data.plot(column=var, legend=True, cmap=cmap, figsize=(10, 5), legend_kwds={'label': "prix m2 moyen des biens de l'IRIS"})
    figure.set_axis_off()

    if output_dir:
      filename = os.path.join(output_dir, f"prix m2 moyen des biens de l'IRIS, {area}.png")
      plt.savefig(filename)
    else:
      plt.show()
  except ValueError as e:
    print(f"Error: {e}")
  except Exception as e:
    print(f"An error occurred while plotting the variable: {e}")


def bien_prix_m2(commune, data, area, output_dir=None):
  """
  Plot all the properties of an area on top of the 'commune'. 
  Color corresponds to prix_m2.

  Args:
    commune (GeoDataFrame): GeoDataFrame representing the commune.
    data (geopandas.GeoDataFrame): The GeoDataFrame containing property information.
    area (str): the name of the area to plot (e.g. "Paris", "Marseille", "Lyon")  - area (str): name of the area of interest.
    output_dir (str, optional): output directory to save the plot.

  Returns:
    None

  Raises:
    ValueError: If the area is not found in the data DataFrame.
  """
  try:
    print(f"Generating 'prix m2' des biens for {area}...")

    cmap = plt.get_cmap("jet")
    data.to_crs(commune.crs)
    area_data = data[data['NOM_COM'] == area]

    if area_data.empty:
      raise ValueError(f"No data found for {area}")

    base = commune[commune.nom == area].plot(figsize=(10, 5), alpha=0.1)
    figure = area_data.plot(ax=base,
                            column='prix_m2_actualise',
                            figsize=(10, 5),
                            alpha=0.2,
                            legend=True,
                            cmap=cmap,
                            legend_kwds={'label': "prix m2 des biens"})
    figure.set_axis_off()

    if output_dir:
      filename = os.path.join(output_dir, f"prix m2 des biens de l'IRIS, {area}.png")
      plt.savefig(filename)
    else:
      plt.show()
  except Exception as e:
      print(f"Error occurred while plotting properties: {e}")


def iris_bien(data, iris, area, output_dir=None):
  """
  Plot all the properties in a area on the background of the IRIS.
  Colors correspond to the median income (IRIS) and the price_m2 of each property.

  Args:
      data (geopandas.GeoDataFrame): The GeoDataFrame containing property information.
      iris (geopandas.GeoDataFrame): The GeoDataFrame containing IRIS information.
      area (str): the name of the area to plot (e.g. "Paris", "Marseille", "Lyon")
      output_dir (str): Optional directory path to save the output plot.

  Returns:
      None.
  """

  try:
    print(f"Generating `revenus médian` et `prix_m2` for {area}...")

    cmap = plt.get_cmap("jet")
    data.to_crs(iris.crs)
    base = iris.loc[iris.NOM_COM == area].plot(column='DISP_MED19',
                                              figsize=(10, 5),
                                              cmap=cmap,
                                              legend=True,
                                              legend_kwds={'label': "revenu médian dans l'IRIS"})
    figure = data[data.NOM_COM == area].plot(ax=base,
                                                column='prix_m2_actualise',
                                                figsize=(10, 5),
                                                legend=True,
                                                cmap=cmap,
                                                legend_kwds={'label': "prix m2 des biens"})
    figure.set_axis_off()

    if output_dir is not None:
      output_file = os.path.join(output_dir, f"revenu médian_vs_prix m2 des biens_{area}.png")
      plt.savefig(output_file)
    else:
      plt.show()
  except Exception as e:
    print(f"Error occurred while plotting: {e}")


def iris_bien_moyen(data, iris, area, metrique, var_iris, name_var_iris, output_dir=None):
  """
  Plot the average of a given metric for each IRIS in a specific area, along with the corresponding values of a
  variable for each IRIS, on the map of the area. The colors on the map correspond to the average metric values
  (per IRIS) and the variable values (per IRIS).

  Args:
    data (geopandas.GeoDataFrame): The GeoDataFrame containing property information.
    iris (geopandas.GeoDataFrame): The GeoDataFrame containing IRIS information.
    area (str): the name of the area to plot (e.g. "Paris", "Marseille", "Lyon")
    metrique (str): the name of the metric to use for the color scale on the map
    var_iris (str): the name of the variable to use for the color scale on the map
    name_var_iris (str): a display name for the variable, to use in the color scale label on the map
    output_dir (str): optional, the path to a directory where the resulting plot will be saved

  Returns:
    None
  """

  try:
    print(f"Generating `{var_iris}` et `prix_m2` for {area}...")

    cmap = plt.get_cmap("jet")

    inter = data.groupby(['DCOMIRIS'])[[metrique]].mean()
    inter = inter.reset_index()
    iris_moyenne = inter.merge(iris, how='right', left_on='DCOMIRIS', right_on='DCOMIRIS')
    iris_moyenne = gpd.GeoDataFrame(iris_moyenne[[metrique, 'DCOMIRIS', 'NOM_COM']], geometry=iris_moyenne['geometry'])

    iris_moyenne['center'] = iris_moyenne.centroid
    
    if iris_moyenne.crs is None:
      iris_moyenne = iris_moyenne.set_crs(epsg=2154) # sets the initial crs of the dataframe
      iris_moyenne = iris_moyenne.to_crs(str(iris.crs))
      iris_moyenne = iris_moyenne.set_geometry('center').to_crs(str(iris.crs))

    base = iris.loc[iris.NOM_COM == area].plot(column=var_iris,
                                                figsize=(10, 5),
                                                legend=True,
                                                cmap=cmap,
                                                legend_kwds={'label': f"{name_var_iris}. dans l'IRIS"})
    figure = iris_moyenne[iris_moyenne.NOM_COM == area].plot(ax=base,
                                                            column=metrique,
                                                            figsize=(10, 5),
                                                            legend=True,
                                                            cmap=cmap,
                                                            legend_kwds={'label': f"{metrique}. dans l'IRIS"})
    figure.set_axis_off()

    if output_dir:
      plt.savefig(os.path.join(output_dir, f"{area}_{metrique}_{name_var_iris}.png"))
    else:
      plt.show()
  except Exception as e:
    print(f"Error occurred while plotting data: {e}")

def plot_equi_commune(equipements, communes, area, num_com, output_dir=None):
  """
  Plot the equipements in a given 'commune'.

  Args:
    equipements (geopandas.GeoDataFrame): The GeoDataFrame containing equipements information.
    communes (geopandas.GeoDataFrame): The GeoDataFrame containing communes information.
    area (str): Name of the commune to plot.
    num_com (str): Departmental code of the commune.
    size (tuple): Figure size (width, height).
    output_file (str): Path to output file to save the plot. Defaults to None.

  Returns:
    None
  """

  try:
    print(f"Generating 'equipements' distribution in {area}_{num_com}...")

    # Create a base map of the commune
    base = communes[communes.nom == area].plot(figsize=(10, 5), alpha=0.1)

    # Plot the equipements on the base map
    figure = equipements[equipements['DEPCOM'] == num_com].plot(ax=base,
                                                            column='TYPEQU',
                                                            figsize=(10, 5),
                                                            legend=True,
                                                            cmap=plt.get_cmap("jet")
                                                            # legend_kwds={'label': "Type d'équipement"}
                                                            )
    # Hide axis
    figure.set_axis_off()

    # Save plot to output file if path is provided
    if output_dir:
      output_file = os.path.join(output_dir, f"distribution equipements_{area}_{num_com}.png")
      plt.savefig(output_file)
    else:
      plt.show()

  except KeyError:
    print(f"Commune '{area}' not found in shapefile")
  except Exception as e:
    print(f"Error occurred while plotting data: {e}")


def plot_equi_iris(equipements, iris, num_iris, output_dir=None):
  """
  Plots a choropleth map showing the distribution of different types of amenities in a given IRIS.

  Args:
    equipements (geopandas.GeoDataFrame): The GeoDataFrame containing equipements information.
    iris (geopandas.GeoDataFrame): The GeoDataFrame containing IRIS information.
    num_iris (str): the code of the IRIS to plot the amenities distribution for
    size (tuple): the size of the plot
    output_dir (str): the directory to save the plot image to (default is None)

  Returns:
    None
  """
  print(f"Generating 'equipements' distribution in {num_iris}...")

  try:
      # plot the IRIS base map
    base = iris[iris.DCOMIRIS == num_iris].plot(figsize=(10, 5), alpha=0.1)

    # plot the choropleth map
    figure = equipements[equipements['DCIRIS'] == num_iris].plot(ax=base,
                                                                        column='TYPEQU',
                                                                        figsize=(10, 5),
                                                                        legend=True,
                                                                        cmap=plt.get_cmap("jet"))

    figure.set_axis_off()

    # save the plot to output directory if given
    if output_dir:
      plt.savefig(os.path.join(output_dir, f"distribution equipements_{num_iris}.png"))
    else:
      plt.show()

  except ValueError as e:
    print(f"Error occurred while plotting the map: {e}")
  except Exception as e:
    print(f"Error occurred: {e}")


def corr_iris(data:gpd.GeoDataFrame, method:str, iris:list, col_1:str, col_2:str) -> float:
    """
    Computes the correlation between two columns of a dataset for a given IRIS region.

    Args:
      data (gpd.GeoDataFrame): input GeoDataFrame
      method (str): the correlation method to use (pearson, spearman, kendall)
      iris (list): the IRIS region to consider
      col_1 (str): the name of the first column
      col_2 (str): the name of the second column

    Returns:
      corr (float): the computed correlation

    Raises:
      ValueError: if the provided method is not supported
      ValueError: if the provided columns are not found in the dataset
    """

    # Select data for the given IRIS region
    points_iris = data[(data['IRIS_x']==iris[0]) & (data['IRIS_y']==iris[1])]
  
    # Select the two columns to compute correlation for
    try:
        x, y = [points_iris[col_1],points_iris[col_2]]
    except KeyError:
        raise ValueError(f"The provided columns '{col_1}' and '{col_2}' are not found in the dataset.")
  
    # Compute correlation if conditions are met
    if (len(x)>2) & (np.min(x)!= np.max(x)) & (np.min(y)!= np.max(y)):
        if method == 'pearson':
            corr = scipy.stats.pearsonr(x, y)[0]
        elif method == 'spearman':
            corr = scipy.stats.spearmanr(x, y)[0]
        elif method == 'kendall':
            corr = scipy.stats.kendalltau(x, y)[0]
        else:
            raise ValueError(f"The correlation method '{method}' is not supported.")
    else:
        corr = np.nan
  
    return corr


def plot_corr_spatiale(data, iris, communes, area, col_1='prix_m2_actualise', col_2='prix_m2_zone', 
                      method='spearman', output_dir=None):
    """
    Plot the spatial correlation of two columns of a specific area.

    Args:
      data (gpd.GeoDataFrame): input GeoDataFrame
      iris (geopandas.GeoDataFrame): The GeoDataFrame containing IRIS information.
      area (str): name of the area to plot the correlation
      col_1 (str): the name of the first column to calculate the correlation
      col_2 (str): the name of the second column to calculate the correlation
      method (str): the method to use for calculating the correlation, defaults to 'spearman'
      output_dir(str, optional: the directory to save the plot image to (default is None)

    Returns:
      None

    """

    try:
      print(f"Plotting correlation for: {area} using {method} correlation..")
      print(f"Note: you can specify 'pearson' or 'kendall 'correlations as arguments.")


      # Calculate the spatial correlation for each iris in the area
      base_area = iris.loc[iris.NOM_COM == area].reset_index(drop=True)
      column_corr = []
      for index_iris in tqdm(base_area.index):
        data_iris = base_area.iloc[index_iris]
        column_corr.append(corr_iris(data, method, [data_iris['IRIS_x'], data_iris['IRIS_y']], col_1, col_2))
      base_area['corr_spatiale'] = column_corr

      # Plot the spatial correlation on the map
      base = communes[communes.nom == area].plot(figsize=(10, 5), alpha=0.1)
      figure = base_area.plot(ax=base,
                            column='corr_spatiale',
                            figsize=(10, 5),
                            legend=True,
                            cmap=plt.get_cmap("jet"),
                            legend_kwds={'label': "Corrélation spatiale dans l'IRIS ({}) dans la ville de {}".format(method, area)})
      figure.set_axis_off()
      # save the plot to output directory if given
      if output_dir:
        plt.savefig(os.path.join(output_dir, f"correlation_spatiale_{area}.png"))
      else:
        plt.show()
    except KeyError:
      print(f"The area '{area}' is not available.")
    except Exception as e:
      print(f"An error occurred while plotting the spatial correlation: {e}")
