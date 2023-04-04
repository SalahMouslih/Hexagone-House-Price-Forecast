import geopandas as gpd
import scipy.stats
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
from numpy import mean
import glob
import os
from datetime import datetime
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def distribution_target_type_and_metropoles(data):
  for met in data.LIBEPCI.unique():
    g = sns.displot(data = data[data.LIBEPCI == met], x="prix_m2_actualise", hue="type_local", kde=True).set(title='Distribution of prix_m2_actualise for flats and houses, {}'.format(met))


def plot_heatmap(data, variable_ref, local=None, color= sns.diverging_palette(29 , 29, center = 'light', sep = 1, n = 40)):
  plt.figure(figsize=(8, 12))
  if local:
    heatmap = sns.heatmap(data[data.LIBEPCI == local].corr()[[variable_ref]].sort_values(by=variable_ref, ascending=False), vmin=-1, vmax=1, annot=True, cmap=color)
    heatmap.set_title("Features Correlating with {}, {}".format(variable_ref, local), fontdict={'fontsize':18}, pad=16);  
  else:
    heatmap = sns.heatmap(data.corr()[[variable_ref]].sort_values(by=variable_ref, ascending=False), vmin=-1, vmax=1, annot=True, cmap=color)
    heatmap.set_title("Features Correlating with {}".format(variable_ref), fontdict={'fontsize':18}, pad=16);

def plot_flats_houses_shares(data):
  
  # set the figure size
  plt.figure(figsize=(20, 8))

  # from raw value to percentage
  total = data.groupby('LIBEPCI')['type_local'].count().reset_index()
  type_local = data[data.type_local=='Appartement'].groupby('LIBEPCI')['type_local'].count().reset_index()
  type_local['type_local'] = [i / j * 100 for i,j in zip(type_local['type_local'], total['type_local'])]
  total['type_local'] = [i / j * 100 for i,j in zip(total['type_local'], total['type_local'])]

  # bar chart 1 -> top bars (group of 'smoker=No')
  bar1 = sns.barplot(x="LIBEPCI",  y="type_local", data=total, color='orange')

  # bar chart 2 -> bottom bars (group of 'smoker=Yes')
  bar2 = sns.barplot(x="LIBEPCI", y="type_local", data=type_local, color='grey')

  # add legend
  top_bar = mpatches.Patch(color='orange', label='Maison')
  bottom_bar = mpatches.Patch(color='grey', label='Appartement')
  plt.legend(handles=[top_bar, bottom_bar])

  # show the graph
  plt.xticks(rotation=40)
  plt.show()