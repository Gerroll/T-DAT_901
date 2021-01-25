import numpy as np
import pandas as pd
import json
from datetime import datetime
from hdbscan import HDBSCAN
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import hdbscan

# Adjust pandas console display
pd_width = 320
pd.set_option('display.width', pd_width)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent
# path to processed data
proc_data_dir = project_dir.joinpath("assets").joinpath('processed')
segmentation_proc_file = proc_data_dir.joinpath("segmentation_proc.pkl")
segmentation_cluster_file = proc_data_dir.joinpath("segmentation_cluster.pkl")
segmentation_cluster_middle_file = proc_data_dir.joinpath("segmentation_cluster_middle.pkl")
segmentation_result_file = proc_data_dir.joinpath("segmentation_result.csv")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")

"""
    Features
"""
amount = ['N_PURCHASE', 'T_PRICE', 'N_BASKET']
familles = ['HYGIENE', 'SOINS DU VISAGE', 'PARFUMAGE', 'SOINS DU CORPS', 'MAQUILLAGE', 'CAPILLAIRES',
                    'SOLAIRES', 'MULTI FAMILLES', 'SANTE NATURELLE']
mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


class Processor:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        # At the end of first processing, data processed dataframe are saved into pickle file
        if segmentation_proc_file.is_file():
            self.__load_file()
        else:
            self.__process()

    def get_raw_data(self):
        return self.__raw_df

    def get_processed_data(self):
        return self.__data

    def __load_file(self):
        self.__data = pd.read_pickle(segmentation_proc_file)

    def pca_component(self, n_components: int, df: pd.DataFrame, show_explained_var: bool):
        # Standardize the data to have a means of ~0 and a variance of 1
        x_std = StandardScaler().fit_transform(df)

        # Create a PCA instance: pca
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(x_std)

        if show_explained_var:
            # Plot the explained variances
            features = range(pca.n_components_)
            plt.bar(features, pca.explained_variance_ratio_, color='black')
            plt.xlabel('PCA features')
            plt.ylabel('variance %')
            plt.xticks(features)
            plt.show()

        # Save components to a DataFrame
        pca_components = pd.DataFrame(principal_components)
        evr = pca.explained_variance_ratio_
        print(len(evr))
        print(sum(evr))
        return pca_components

    def looking_for_most_relevant_n_cluster(self, td, feat=None, max_num_cluster=20):
        if feat is None:
            df = td
        else:
            df = pd.DataFrame()
            for f in feat:
                df[f] = td[f]
        # compute inertia
        inertias = []
        ks = range(1, max_num_cluster)
        for k in ks:
            print(k)
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=k)

            # Fit model to samples
            model.fit(df)

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.show()

    def optimal_eps_DBSCAN(self, df, n_neighbors):
        df = StandardScaler().fit_transform(df)
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(df)
        distances, indices = nbrs.kneighbors(df)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.show()

    def __process(self):
        # Retrieve all unique client ID
        client_ids = self.__raw_df["CLI_ID"].unique()
        df_col = ['CLI_ID']
        ticket_ids = set()

        df_col.extend(amount)
        df_col.extend(familles)
        df_col.extend(mois)

        # init collecting
        print(f"Init collecting at {datetime.now().time()}")
        collect = {k: [0] * 24 for k in client_ids}

        # collect data
        print(f"Start collecting at {datetime.now().time()}")
        for row in self.__raw_df.itertuples():
            self.__complete_collecting(collect, row, ticket_ids)

        # normalize mois and famille
        print("normalize mois and famille")
        for key, val in collect.items():
            for i in range(3, 24):
                collect[key][i] = round(collect[key][i] * 100 / collect[key][0], 2)

        # build result
        print(f"Start building dataframe at {datetime.now().time()}")
        npa = [[key] + value for key, value in collect.items()]
        self.__data = pd.DataFrame(
            np.array(npa),
            columns=df_col
        )
        print(f"End preprocess at {datetime.now().time()}")

        # Save to pickle
        self.__data.to_pickle(segmentation_proc_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation_proc.pkl")

    def __complete_collecting(self, collect, row, ticket_ids):
        ticket_id = row[1]
        mois_index = int(row[2]) - 1
        prix_net = round(float(row[3]), 2)
        famille = row[4]
        cli_id = row[8]
        famille_index = familles.index(famille)

        # number of purchase
        collect[cli_id][0] += 1
        # sum of prices
        collect[cli_id][1] += prix_net
        # number of baskets
        if ticket_id not in ticket_ids:
            collect[cli_id][2] += 1
            ticket_ids.add(ticket_id)

        # famille proportion
        collect[cli_id][3 + famille_index] += 1

        # mois proportion
        collect[cli_id][12 + mois_index] += 1


class Clusterer:
    def __init__(self):
        proc = Processor()
        self.__raw_df = proc.get_raw_data()
        self.__data = proc.get_processed_data()
        self.__cli_size = len(self.__data['CLI_ID'].unique())

        #
        # This part is use too cluster only mois and cluster family
        #
        if segmentation_cluster_middle_file.is_file():
            self.__load_middle_predicted_data()
        else:
            self.__remake_middle_prediction()

        #
        # final clustering
        #
        if segmentation_cluster_file.is_file():
            self.__load_predicted_data()
        else:
            self.__remake_prediction()

    def get_predicted_data(self):
        return self.__data

    def __load_predicted_data(self):
        self.__data = pd.read_pickle(segmentation_cluster_file)

    def __load_middle_predicted_data(self):
        self.__data = pd.read_pickle(segmentation_cluster_middle_file)

    def __remake_prediction(self):
        # format data
        df = pd.DataFrame()
        features = ['N_PURCHASE', 'T_PRICE', 'N_BASKET', 'C_MOIS', 'C_FAMILLE']
        for col in features:
            df[col] = self.__data[col]
        df = StandardScaler().fit_transform(df)

        # fit algorithm
        print("start fitting final")
        min_cluster_size = 12650
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
        clusterer.fit(df)
        print(f"Number of labels: {clusterer.labels_.max() + 1}")
        self.__data['cluster'] = clusterer.labels_

        # save prediction
        self.__data.to_pickle(segmentation_cluster_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation_cluster.pkl")

    def __remake_middle_prediction(self):
        #
        # FAMILLES
        #
        df = pd.DataFrame()
        for col in familles:
            df[col] = self.__data[col]
        df = StandardScaler().fit_transform(df)

        # fit algorithm
        print("start fitting familles")
        # 6 find from elbow method
        num_cluster = 6
        clusterer = KMeans(n_clusters=num_cluster)
        clusterer.fit(df)
        self.__data['C_FAMILLE'] = clusterer.labels_

        #
        # MOIS
        #
        df2 = pd.DataFrame()
        for col in mois:
            df2[col] = self.__data[col]
        df2 = StandardScaler().fit_transform(df2)

        # fit algorithm
        print("start fitting mois")
        # 13 find from elbow method
        num_cluster = 13
        clusterer2 = KMeans(n_clusters=num_cluster)
        clusterer2.fit(df2)
        self.__data['C_MOIS'] = clusterer2.labels_

        # save prediction
        self.__data.to_pickle(segmentation_cluster_middle_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation_cluster_middle.pkl")

    def __p2percent(self, p):
        return round(p * 100, 2)

    def get_description(self, cluster_n):
        cluster = self.__filter_by_cluster_label(cluster_n)

        # df size
        size = len(cluster)

        # client proportion
        t_cli = len(cluster['CLI_ID'].unique())
        p = t_cli / self.__cli_size
        c_proportion = self.__p2percent(p)

        # famille proportion
        f_proportions = []
        for f in familles:
            fam_cluster = cluster[cluster['FAMILLE'] == f]
            p = len(fam_cluster) / size
            f_proportions.append(self.__p2percent(p))

        # mois proportion
        m_proportions = []
        for i in range(1, 13):
            mois_cluster = cluster[cluster['MOIS_VENTE'] == i]
            p = len(mois_cluster) / size
            m_proportions.append(self.__p2percent(p))

        # price proportion
        sum_price = cluster['PRIX_NET'].sum()
        mean_price = round(sum_price / t_cli, 2)

        # basket proportion
        n_basket = len(cluster['TICKET_ID'].unique())
        p = n_basket / t_cli
        mean_basket = round(p, 2)

        target = {
            'client_proportion': c_proportion,
            'mean_expense': mean_price,
            'mean_basket': mean_basket,
            'proportion_purchase_by_family': f_proportions,
            'proportion_purchase_by_month': m_proportions
        }
        print(json.dumps(target, indent=2))
        return target

    def __filter_by_cluster_label(self, cluster_n):
        df = self.__data
        cluster = df[df['cluster'] == cluster_n]
        list_id = set(cluster['CLI_ID'])
        return self.__raw_df[self.__raw_df['CLI_ID'].isin(list_id)]

    def __special_print(self, target):
        for key, value in target.items():
            print(key)
            print(value)

    def to_csv(self):
        cluster_labels = sorted(list(self.__data['cluster'].unique()))
        print(f"cluster labels: {cluster_labels}")
        print(f"Proportion of clients who did not find a cluster: {len(self.__data[self.__data['cluster'] == -1])} / {len(self.__data)}")
        cluster_labels.remove(-1)
        rows = []
        familles_col = [f + " (%)" for f in familles]
        mois_col = [m + " (%)" for m in mois]
        columns = ['cluster', 'client_proportion (%)', 'mean_expense', 'mean_basket'] + familles_col + mois_col
        for i in cluster_labels:
            print(i)
            desc = clust.get_description(i)
            row = []
            row.extend([str(i), desc['client_proportion'], desc['mean_expense'], desc['mean_basket']])
            row.extend(desc['proportion_purchase_by_family'])
            row.extend(desc['proportion_purchase_by_month'])
            rows.append(row)

        pd.DataFrame(
            np.array(rows),
            columns=columns
        ).to_csv(segmentation_result_file)


if __name__ == "__main__":
    clust = Clusterer()
    clust.to_csv()
