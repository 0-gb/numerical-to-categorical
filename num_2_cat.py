import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from sklearn.cluster import KMeans
from matplotlib import colors
from sklearn.preprocessing import OneHotEncoder


def get_cluster_count(input_df_column, max_clusters):
    cluster_costs = []
    for cluster_count in range(1, max_clusters):
        k_means_plot = KMeans(n_clusters=cluster_count)
        k_means_plot.fit_predict(input_df_column)
        cluster_costs.append(k_means_plot.inertia_)
    cluster_costs_log = [log(element) for element in cluster_costs]
    plt.plot(cluster_costs_log)
    plt.show()
    return int(input("Enter the identified number of clusters:\n"))


def numerical_2_categorical(input_df_column, max_clusters=20, cluster_count=-1, hist_bins=100, y_max=1):
    if cluster_count < 0:
        cluster_count = get_cluster_count(input_df_column, max_clusters)
    print("Working with " + str(cluster_count) + " clusters.")

    # k means for clustering
    k_means = KMeans(n_clusters=cluster_count)
    labeled = k_means.fit_predict(input_df_column)

    # Make a histogram with the provided data
    fig, ax = plt.subplots()
    _, bins, patches = ax.hist(pd.DataFrame(input_df_column), bins=hist_bins,  edgecolor='white', linewidth=1)

    # Mark the cluster centers
    for element in k_means.cluster_centers_:
        ax.axvline(x=element, ymin=0.0, ymax=y_max, color='red')

    # Define the colors on the histogram
    hist_colors = random.sample(list(colors.CSS4_COLORS), cluster_count)  # select random colors for clusters
    ax.set_facecolor((0.0, 0.0, 0.0))  # set the background to be black for better visibility of clusters
    for i in range(0, len(patches)):  # define each cluster color on the histogram
        diffs = abs((bins[i]+bins[i+1])/2 - k_means.cluster_centers_)
        selected_color = hist_colors[np.argmin(diffs)]
        patches[i].set_facecolor(selected_color)
    plt.show()

    # Turn the clusters into one hot encoded values
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_matrix = encoder.fit_transform(labeled.reshape(-1, 1)).todense()
    return pd.DataFrame(encoded_matrix)



