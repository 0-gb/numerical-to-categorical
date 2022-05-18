import random
import num_2_cat
import pandas as pd

def generate_data(cluster_count=5, cluster_range=300, seed=0, n_points=50000, mu=0, sigma=10):
    random.seed(seed)
    cluster_centers = [cluster_range*random.random() for _ in range(cluster_count)]
    data_points = [random.choice(cluster_centers)+random.gauss(mu, sigma) for _ in range(n_points)]
    return pd.DataFrame(data_points)


test_df = generate_data()
categorization_result = num_2_cat.numerical_2_categorical(test_df)
input("Finished. Press Enter.")
