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
from math import pi

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
        self.__cluster_analysis = None

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
        clusterer = KMeans(n_clusters=num_cluster, random_state=1)
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
        clusterer2 = KMeans(n_clusters=num_cluster, random_state=1)
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
        if segmentation_result_file.is_file():
            self.__cluster_analysis = pd.read_csv(segmentation_result_file)
            return
        cluster_labels = sorted(list(self.__data['cluster'].unique()))
        print(f"cluster labels: {cluster_labels}")
        print(f"Proportion of clients who did not find a cluster: {len(self.__data[self.__data['cluster'] == -1])} / {len(self.__data)}")
        cluster_labels.remove(-1)
        rows = []
        familles_col = [f + " (%)" for f in familles]
        mm = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
             "novembre", "décembre"]
        mois_col = [m + " (%)" for m in mm]
        columns = ['cluster', 'proportion de la clientèle (%)', 'moyenne des dépenses', 'taille moyenne du panier'] + familles_col + mois_col
        for i in cluster_labels:
            print(i)
            desc = self.get_description(i)
            row = []
            row.extend([str(i), desc['client_proportion'], desc['mean_expense'], desc['mean_basket']])
            row.extend(desc['proportion_purchase_by_family'])
            row.extend(desc['proportion_purchase_by_month'])
            rows.append(row)

        self.__cluster_analysis = pd.DataFrame(np.array(rows), columns=columns)
        self.__cluster_analysis.to_csv(segmentation_result_file)

    def radar_chart_all_data(self, type_t):
        analysis = []
        for column in self.__cluster_analysis:
            serie = self.__cluster_analysis[column].astype(float)
            m = -1
            if type_t == "mean":
                m = serie.mean()
            elif type_t == "min":
                m = serie.min()
            elif type_t == "max":
                m = serie.max()
            analysis.append(m)
        return analysis

    def radar_chart_one_data(self, cluster_label):
        return list(self.__cluster_analysis.iloc[cluster_label])

    def __remarquable_general(self, remarquables, type_t):
        description = f"Caractéristiques remarquables {type_t}:\n"
        up = []
        down = []

        for remarquable in remarquables:
            value = remarquable["value"]
            feat = remarquable["feature"]
            if remarquable["ext"] == 1:
                up.append((feat, value))
            elif remarquable["ext"] == 0:
                down.append((feat, value))
        if len(up) != 0:
            description += f" HAUTES:\n"
            for cat in up:
                description += f"     -{cat[0]}: {cat[1]}\n"
            description += "\n"
        if len(down) != 0:
            description += f" BASSES:\n"
            for cat in down:
                description += f"     -{cat[0]}: {cat[1]}\n"
        return description

    def display_radar(self, data, remarquables, cluster_label):

        # ------- PART 1: Create background

        categories_label = [
            ["hygiene", "soins du visage", "parfumage", "soins du corps", "maquillage", "capillaires", "solaires"],
            ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
             "novembre", "décembre"]
        ]

        clust_prop = data[0][3][2]
        clust_size = len(self.__data[self.__data["cluster"] == cluster_label])
        clust_expense = data[0][3][3]
        clust_basket = data[0][3][4]

        plt.figure().suptitle(f"Cluster {cluster_label}")

        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])

        plt.text(0.04, 0.95, f"Taille: {clust_size} ({clust_prop}% de la clientelle)", size=10)
        plt.text(0.04, 0.9, f"Dépense moyenne: {clust_expense}", size=10)
        plt.text(0.04, 0.85, f"Panier moyen: {clust_basket}", size=10)

        plt.text(0.04, 0.7, self.__remarquable_general(remarquables[0], "générales"), size=10)
        plt.text(0.04, 0.4, self.__remarquable_general(remarquables[1], "familles"), size=10)
        plt.text(0.04, 0, self.__remarquable_general(remarquables[2], "mois"), size=10)

        for i in range(2):
            labels = categories_label[i]
            data_list = data[i + 1]

            N = len(labels)

            # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Initialise the spider plot
            ax = plt.subplot(2, 2, 2 + i * 2, polar=True)

            # If you want the first axis to be on top:
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)

            # Draw one axe per variable + add labels labels yet
            plt.xticks(angles[:-1], labels)

            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90], ["10", "20", "30", "40", "50", "60", "70", "80", "90"], color="grey", size=7)

            # compute maximum
            pot_max = max(data_list[3])
            i_pot_max = data_list[3].index(pot_max)
            other_max = data_list[1][i_pot_max]
            mmax = (pot_max + other_max) / 2
            plt.ylim(0, mmax)

            # ------- PART 2: Add plots

            # Plot each individual = each line of the data
            # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

            # Min
            values = data_list[0]
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label="Min", color="#0000FF")

            # Max
            values = data_list[1]
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label="Max", color="#FF0000")

            # Moyenne
            values = data_list[2]
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label="Moyenne", color="#FF00FF")

            # Cluster
            values = data_list[3]
            values += values[:1]
            ax.fill(angles, values, label="Cluster " + str(cluster_label), color="#000000", alpha=0.4)

            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

    def __compute_remarquable(self, cluster_label):
        remarquables = [[], [], []]
        i = 0
        for column in self.__cluster_analysis:
            serie = sorted(list(self.__cluster_analysis[column]))
            clust_value = self.__cluster_analysis[column].iloc[cluster_label]
            rank = 15 - serie.index(clust_value)
            if rank <= 3 or rank >= 13:
                if i in [2, 3, 4]:
                    remarquables[0].append(
                        {
                            "feature": column,
                            "value": clust_value,
                            "rank": rank,
                            "ext": 1 if rank <= 3 else 0 if rank >= 13 else -1
                        }
                    )
                if 5 <= i <= 11:
                    remarquables[1].append(
                        {
                            "feature": column,
                            "value": clust_value,
                            "rank": rank,
                            "ext": 1 if rank <= 3 else 0 if rank >= 13 else -1
                        }
                    )
                if i >= 14:
                    remarquables[2].append(
                        {
                            "feature": column,
                            "value": clust_value,
                            "rank": rank,
                            "ext": 1 if rank <= 3 else 0 if rank >= 13 else -1
                        }
                    )
            i = i + 1
        return remarquables

    def radar(self, cluster_label):
        remarquables = self.__compute_remarquable(cluster_label)
        min_data = self.radar_chart_all_data("min")
        max_data = self.radar_chart_all_data("max")
        mean_data = self.radar_chart_all_data("mean")
        one_data = self.radar_chart_one_data(cluster_label)
        data = [
            [min_data[:5], max_data[:5], mean_data[:5], one_data[:5]],
            [min_data[5:12], max_data[5:12], mean_data[5:12], one_data[5:12]],
            [min_data[14:], max_data[14:], mean_data[14:], one_data[14:]]
        ]

        self.display_radar(data, remarquables, cluster_label)

    def display_segmentation(self, client_id):
        # the cluster of the client
        self.to_csv()
        list_label = list(self.__data[self.__data["CLI_ID"] == client_id]["cluster"])
        if len(list_label) == 0:
            print("Client's ID doesn't exist")
            return
        cluster_label = list_label[0]
        if cluster_label == -1:
            print("Client's ID can't find a pertinent segmentation")
            return

        self.radar(cluster_label)


if __name__ == "__main__":
    pertinent = [1490281, 13290776, 20163348, 20200041, 20561854, 20727324, 21046542, 21497331, 69813934, 100064590,
                 169985247, 300240190, 336948609, 359489151, 360340862, 800115293]
    clust = Clusterer()

    #clust.display_segmentation(1490281)

    for my_id in pertinent:
        clust.display_segmentation(my_id)
