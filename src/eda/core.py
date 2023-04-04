"""
EDA Core functions

This module contains core functions for visualizing and analyzing processed data.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from numpy import mean
import glob
import os
from datetime import datetime
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
