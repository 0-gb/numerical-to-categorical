# numerical-to-categorical
Turn numerical features into categorical using a one column pandas dataframe. 

The transformation from a numerical to a categorical feature is performed by clustering. The user is to pick the number of clusters in the data. The conventional k-keams process is then used and one hot encoding is performed on the obtained clusters. 

The user is provided with aid in deciding how many clusters to use. This is achieved by checking the log of the residual cost of the k means clustering for an increased number of clusters: when the number of clusters is sufficiently large and the added cluster parameter of the k-means isn't doing much, this log of the cost function tends to start decreasing more linearly. The user can use the plot get an insight into the number of clusters.
