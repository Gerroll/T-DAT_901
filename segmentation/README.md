### Segmentation

This module provides client description from profiling using a process of clustering.

##### Investigate work

In order to reach the goal: find some client profile (between 4 and 20, NOT MUCH: because we assume that is not interesting
for an enterprise to design more than 20 client profile and a market strategie with, NOT LESS: because fewer profiles induces 
a too poor description of data and its diversity).

In order to process thousands or billions rows dataset with many features, we decide to experiment clustering algorithms
from essentially scikit-learn library.

The setting of algorithm and preprocessing was the main work to give our data the best shape to keep the maximum of the information 
but with the less complexity. We experiment here three types of algorithm: centroid-based, density-based, hierarchical-based.
Among these areas some algorithms seemed to be the most suitable for our data (visualization, dataset size, execution time , number
of feature). We choose to use KMeans (centroid-based) and hdbscan (density-based AND hierarchical-based)

##### Preprocessing

KMeans exposes many advantages, efficient, fast, good with many features data and regroup ALL data, on the other hand it's sensitives
to initial conditions and outliers and produces a coarse clustering.

Hdbscan, it is more or less the opposite, he detects outliers and excludes them, he is slower (but fastest than DBSCAN and a majority
of scikit-learn algorithms), finer clustering based on density but not good with big size features.

So the goal was to combine their uses to produce users data with the most complete description.

First the original dataset was processed to produce the following data:

    CLIENT_ID | NUMBER OF PURCHASE | SUM OF EXPENSES | AVERAGE BASKET | PROPORTION OF TRANSACTIONS CARRIED OUT BY FAMILY (9 features) | PROPORTION OF TRANSACTIONS CARRIED OUT BY MONTH (12 features)

At this point, the problem was this data contains 24 features (without CLIENT_ID) which was difficult to process (which take between
3 and 12 hours to cluster) and the worst, not revealing for many settings of HDBSCAN.
The trick was to pre-grouping this data on PROPORTION OF TRANSACTIONS CARRIED OUT BY FAMILY and PROPORTION OF TRANSACTIONS CARRIED OUT BY MONTH
with the KMeans algorithm, particularly effective on a data tha contains only proportion by family for example
After this pre-grouping with KMeans, we obtain a new data with only 5 features:

    CLIENT_ID | NUMBER OF PURCHASE | SUM OF EXPENSES | AVERAGE BASKET | CLUSTER_LABEL (FAMILY) | CLUSTER_LABEL (MONTH)

##### Clustering

As explain on the previous section, 2 pre-clustering was made on preprocessed data:
- Kmeans on a data that contains only family proportions
- Kmeans on a data that contains only month proportions
It was performs Elbow method to find the most revelant number of cluster (6 for family, and 13 for months)

On the previous data has been applied a hdbscan algorithm to find a revelant number of cluster (between 5 and 20).
At the end we found 15 customer profiles with a minimum size cluster of 12650, reducing the number of ungrouped customers (???) and increasing 
diversity on each profile.





