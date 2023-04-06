import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load data from CSV file, use first row as column names
data = pd.read_csv('F:/UET/SEMESTER 6/ML/Assignment 4/housing data.csv', header=0)

# select longitude and latitude columns
coords = data[['longitude', 'latitude']]

# initialize KMeans with 6 clusters
kmeans = KMeans(n_clusters=6)

# fit KMeans to data
kmeans.fit(coords)

# get cluster labels for each data point
labels = kmeans.labels_

# add cluster labels to the dataframe
data['Cluster'] = labels

# initialize the seaborn plot
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(8,6))

# plot the scatterplot with different colors representing each cluster
sns.scatterplot(x='longitude', y='latitude', hue='Cluster', data=data, palette='deep', s=20, ax=ax)

# set x and y axis labels
ax.set(xlabel='Longitude', ylabel='Latitude')

# display the plot
plt.show()

