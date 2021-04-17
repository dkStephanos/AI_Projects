import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

white_df = pd.read_csv('winequality-white.csv')
target_white = white_df['quality']
white_data = white_df.drop(columns=['quality'])

white_clusters = KMeans(n_clusters=11).fit(white_data)
white_labels = white_clusters.labels_

clusters = {}
for i in range(11):
    clusters[i] = pd.DataFrame(white_df.values[white_labels == i, :])    # Subset of the datapoints that have been assigned to the cluster i
    print(clusters[i][11].mean())

red_df = pd.read_csv('winequality-red.csv')
target_red = red_df['quality']
red_data = red_df.drop(columns=['quality'])

red_clusters = KMeans(n_clusters=11).fit(red_data)
red_labels = red_clusters.labels_

clusters = {}
for i in range(11):
    clusters[i] = pd.DataFrame(red_df.values[red_labels == i, :])    # Subset of the datapoints that have been assigned to the cluster i
    print(clusters[i][11].mean())