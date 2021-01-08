import numpy as np
import pandas as pd
import json
from datetime import datetime
import hdbscan
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
import matplotlib.pyplot as plt
import sys
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
proc_data_dir = project_dir.joinpath("processed-data")
segmentation3_proc_cluster_file = proc_data_dir.joinpath("segmentation3_proc_cluster.pkl")
segmentation3_proc_file = proc_data_dir.joinpath("segmentation3_proc.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")


class Processor:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        # At the end of first processing, data processed dataframe are saved into pickle file
        if segmentation3_proc_file.is_file():
            self.__load_file()
        else:
            self.__process()

    def get_raw_data(self):
        return self.__raw_df

    def get_processed_data(self):
        return self.__data

    def __load_file(self):
        self.__data = pd.read_pickle(segmentation3_proc_file)

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

    def looking_for_most_relevant_n_cluster(self, df):
        # compute inertia
        inertias = []
        ks = range(1, 20)
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
        mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        familles = ['HYGIENE', 'SOINS DU VISAGE', 'PARFUMAGE', 'SOINS DU CORPS', 'MAQUILLAGE', 'CAPILLAIRES',
                    'SOLAIRES', 'MULTI FAMILLES', 'SANTE NATURELLE']
        amount = ['N_PURCHASE', 'T_PRICE', 'N_BASKET']
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
        self.__data.to_pickle(segmentation3_proc_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation3_proc.pkl")

    def __complete_collecting(self, collect, row, ticket_ids):
        familles = ['HYGIENE', 'SOINS DU VISAGE', 'PARFUMAGE', 'SOINS DU CORPS', 'MAQUILLAGE', 'CAPILLAIRES',
                    'SOLAIRES', 'MULTI FAMILLES', 'SANTE NATURELLE']
        ticket_id = row[1]
        mois_index = int(row[2]) - 1
        prix_net = float(row[3])
        famille = row[4]
        cli_id = row[8]
        famille_index = familles.index(famille)

        # number of purchases
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
        mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        familles = ['HYGIENE', 'SOINS DU VISAGE', 'PARFUMAGE', 'SOINS DU CORPS', 'MAQUILLAGE', 'CAPILLAIRES',
                    'SOLAIRES', 'MULTI FAMILLES', 'SANTE NATURELLE']
        amount = ['N_PURCHASE', 'T_PRICE', 'N_BASKET']
        proc = Processor()
        self.__raw_df = proc.get_raw_data()
        self.__data = proc.get_processed_data()
        self.__cli_size = len(self.__data['CLI_ID'].unique())
        if segmentation3_proc_cluster_file.is_file():
            self.__load_predicted_data()
        else:
            self.__remake_prediction(amount + familles + amount)
        print(set(self.__data['cluster']))

    def get_predicted_data(self):
        return self.__data

    def __load_predicted_data(self):
        self.__data = pd.read_pickle(segmentation3_proc_cluster_file)

    def __remake_prediction(self, features):
        # format data
        ori = self.__data.copy()
        df = pd.DataFrame()
        for col in features:
            df[col] = ori[col]
        df = StandardScaler().fit_transform(df)

        # fit algorithm
        print("start fitting")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
        clusterer.fit(df)
        print(f"Number of labels: {clusterer.labels_.max()}")

        self.__data['cluster'] = clusterer.labels_

        # save prediction
        self.__data.to_pickle(segmentation3_proc_cluster_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation3_proc_cluster.pkl")

    def get_description(self, cluster_n):
        cluster = self.__filter_by_cluster_label(cluster_n)
        self.display_information(cluster)

    def __p2percent(self, p):
        return str(round(p * 100, 2)) + ' %'

    def display_information(self, df):
        # df size
        size = len(df)

        # client proportion
        t_cli = len(df['CLI_ID'].unique())
        p = t_cli / self.__cli_size
        c_proportion = self.__p2percent(p)

        # famille proportion
        familles = ['HYGIENE', 'SOINS DU VISAGE', 'PARFUMAGE', 'SOINS DU CORPS', 'MAQUILLAGE', 'CAPILLAIRES',
                    'SOLAIRES', 'MULTI FAMILLES', 'SANTE NATURELLE']
        f_proportions = []
        for f in familles:
            fam_df = df[df['FAMILLE'] == f]
            p = len(fam_df) / size
            f_proportions.append(self.__p2percent(p))

        # mois proportion
        m_proportions = []
        for i in range(1, 13):
            mois_df = df[df['MOIS_VENTE'] == i]
            p = len(mois_df) / size
            m_proportions.append(self.__p2percent(p))

        # price proportion
        mean_price = df['PRIX_NET'].mean()
        mean_price = round(mean_price, 2)

        # basket proportion
        n_basket = len(df['TICKET_ID'].unique())
        p = n_basket / t_cli
        mean_basket = round(p, 2)

        target = {
            'size': c_proportion,
            'month': m_proportions,
            'famille': f_proportions,
            'mean_price': mean_price,
            'mean_basket': mean_basket

        }
        print(json.dumps(target, indent=2))

    def __filter_by_cluster_label(self, cluster_n):
        df = self.__data
        cluster = df[df['cluster'] == cluster_n]
        list_id = set(cluster['CLI_ID'])
        return self.__raw_df[self.__raw_df['CLI_ID'].isin(list_id)]

    def __special_print(self, target):
        for key, value in target.items():
            print(key)
            print(value)


if __name__ == "__main__":
    proc = Processor()
    df = proc.get_processed_data()
    d = pd.DataFrame()
    amount = ['N_PURCHASE', 'T_PRICE', 'N_BASKET']
    for col in amount:
        d[col] = df[col]

    proc.optimal_eps_DBSCAN(df, 2)

    # clust = Clusterer()
    # p = clust.get_predicted_data()
    # print(len(p))
    # print(len(p[p['cluster'] == -1]))



    # t = pd.DataFrame()
    # for col in amount:
    #     t[col] = df[col]
    # proc.optimal_eps_DBSCAN(t, 2)


    # clusterer = Clusterer()
    # ids = [1490281, 13290776, 20163348, 20200041, 20561854, 20727324, 20791601, 21046542, 21239163,
    #  21351166, 21497331, 21504227, 21514622, 69813934, 71891681, 85057203]
    #
    # for index in ids:
    #     print("")
    #     print("")
    #     print(index)
    #     clusterer.get_description(index)
