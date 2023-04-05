"""
EDA Core functions

This module contains core functions for visualizing and analyzing processed data.
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

def plot_heatmap(data, output_dir=None):
  """
  Plot a heatmap of the correlation matrix between features and "prix_m2_actualise" for a given dataset, 
  either for all metropoles or for a specific metropole, and save the resulting plot to a directory.

  Parameters:
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

  Parameters:
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
  Plot the distribution of prix_m2_actualise for different types of properties (flats and houses) and metropoles, 
  and save the resulting plot to a directory.

  Parameters:
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
      for i, met in enumerate(data.LIBEPCI.unique()):
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

  Parameters:
  data (pandas.DataFrame): Input data containing the columns "LIBEPCI" and "type_local".
  output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. 
  If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
  None
  """

  print("Generating percentage of flats and houses for each metropole...")

  # Set the figure size
  plt.figure(figsize=(20, 8))

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

  Parameters:
  data (pandas.DataFrame): Input data containing the columns 'type_local', 'surface_reelle_bati', and 'nombre_pieces_principales'.
  output_dir (str, optional): Path to the directory where the box plots will be saved as a PNG file. If not provided, the plots will only be displayed and not saved. Defaults to None.

  Returns:
  None
  """

  try:
    print("Generating box plots of 'surface_reelle_bati' and 'nombre_pieces_principales'...")

    fig = plt.figure(figsize=(20, 8))

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

  Parameters:
  data (pandas.DataFrame): Input data containing the columns "type_local", "surface_reelle_bati", and "nombre_pieces_principales".
  output_dir (str, optional): Path to the directory where the figure will be saved. If None, the figure will be displayed using plt.show(). Default is None.

  Returns:
  None
  """

  try:
    print("Generating boxen plots of 'surface_reelle_bati' and 'nombre_pieces_principales'...")

    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(20, 8))
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

  Parameters:
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
  print(f"Generating 'prix m2 moyen' for {area}...")

  """
  Plot a variable for a specific area using the IRIS dataset.

  Args:
  - iris (geopandas.GeoDataFrame): The GeoDataFrame containing IRIS information.
  - area (str): The name of the area to plot.
  - var (str): The name of the variable to plot.
  - output_dir (str, optional): The output directory to save the plot to.

  Returns:
  - None

  Raises:
  - ValueError: If the specified area is not found in the IRIS dataset.
  - Exception: If an error occurs while plotting the variable.

  """
  try:
    cmap = plt.get_cmap("jet")
    data = iris[iris.NOM_COM == area]
    if data.empty:
      raise ValueError(f"The specified area '{area}' was not found in the IRIS dataset.")
    figure = data.plot(column=var, legend=True, cmap=cmap, figsize=(20,10), legend_kwds={'label': "prix m2 moyen des biens de l'IRIS"})
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
  Plot all the properties of an area on top of the commune. Color corresponds to prix_m2.

  Args:
  - commune (GeoDataFrame): GeoDataFrame representing the commune.
  - data (geopandas.GeoDataFrame): The GeoDataFrame containing property information.
  - area (str): the name of the area to plot (e.g. "Paris", "Marseille", "Lyon")  - area (str): name of the area of interest.
  - output_dir (str, optional): output directory to save the plot.

  Returns:
  - None

  Raises:
  - ValueError: If the area is not found in the data DataFrame.

  """
  try:
    print(f"Generating 'prix m2' des biens for {area}...")

    cmap = plt.get_cmap("jet")
    data.to_crs(commune.crs)
    area_data = data[data['NOM_COM'] == area]

    if area_data.empty:
      raise ValueError(f"No data found for {area}")

    base = commune[commune.nom == area].plot(figsize=(20,10), alpha=0.1)
    figure = area_data.plot(ax=base,
                            column='prix_m2_actualise',
                            figsize=(20,10),
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
    print(f"Generating revenus médian et au prix_m2 for {area}...")

    cmap = plt.get_cmap("jet")
    data.to_crs(iris.crs)
    base = iris.loc[iris.NOM_COM == area].plot(column='DISP_MED19',
                                              figsize=(20,10),
                                              cmap=cmap,
                                              legend=True,
                                              legend_kwds={'label': "revenu médian dans l'IRIS"})
    figure = data[data.NOM_COM == area].plot(ax=base,
                                                column='prix_m2_actualise',
                                                figsize=(20,10),
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

  Parameters:
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
    print(f"Generating {var_iris} et au prix_m2 for {area}...")

    cmap = plt.get_cmap("jet")

    inter = data.groupby(['DCOMIRIS_right'])[[metrique]].mean()
    inter.reset_index(inplace=True)
    iris_moyenne = inter.merge(iris, how='right', left_on='DCOMIRIS_right', right_on='DCOMIRIS')
    iris_moyenne = gpd.GeoDataFrame(iris_moyenne[[metrique, 'DCOMIRIS', 'NOM_COM']], geometry=iris_moyenne['geometry'])
    iris_moyenne = iris_moyenne.to_crs(iris.crs)
    
    iris_moyenne['center'] = iris_moyenne.centroid
    iris_moyenne = iris_moyenne.set_geometry('center').to_crs(iris.crs)

    base = iris.loc[iris.NOM_COM == area].plot(column=var_iris,
                                                figsize=(20,10),
                                                legend=True,
                                                cmap=cmap,
                                                legend_kwds={'label': f"{name_var_iris}. dans l'IRIS"})
    figure = iris_moyenne[iris_moyenne.NOM_COM == area].plot(ax=base,
                                                            column=metrique,
                                                            figsize=(20,10),
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
