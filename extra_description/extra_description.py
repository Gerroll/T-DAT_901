import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.cluster import KMeans


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
segmentation_proc_file = proc_data_dir.joinpath("segmentation_proc.pkl")
segmentation_proc_cluster_file = proc_data_dir.joinpath("segmentation_proc_cluster.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")
print(project_dir)
print(data_dir)
print(kado_file)


class Processor:
    def __init__(self,raw_data):
        self.__raw_df = raw_data
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

    def __process(self):
        # Retrieve all unique client ID
        client_ids = self.__raw_df["CLI_ID"].unique()

        # build dataframe column
        df_col = ["CLI_ID", "NUM_BUY", "SUM_PRICE", 'SIZE_BASKET']

        # collect data
        print(f"Start collecting at {datetime.now().time()}")
        collect = {k: [0, 0, set()] for k in client_ids}
        for row in self.__raw_df.itertuples():
            cli_id = row[8]
            collect[cli_id][0] += 1
            collect[cli_id][1] += row[3]
            collect[cli_id][2].add(row[1])

        # build result
        print(f"Start building dataframe at {datetime.now().time()}")
        npa = [[key] + value[:2] + [value[0] / len(value[-1])] for key, value in collect.items()]
        self.__data = pd.DataFrame(
            np.array(npa),
            columns=df_col
        )
        print(f"End preprocess at {datetime.now().time()}")

        # Save to pickle
        self.__data.to_pickle(segmentation_proc_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation_proc.pkl")


class Clusterer:
    def __init__(self,raw_data):
        proc = Processor(raw_data)
        self.__raw_df = proc.get_raw_data()
        self.__data = proc.get_processed_data()
        self.__data_size = len(self.__data)
        if segmentation_proc_cluster_file.is_file():
            self.__load_predicted_data()
        else:
            self.__remake_prediction()

    def get_predicted_data(self):
        return self.__data

    def __load_predicted_data(self):
        self.__data = pd.read_pickle(segmentation_proc_cluster_file)

    def __kmeans_prediction_one_feature(self, feature, n_clusters):
        """
        The number of cluster is determine with Elbow Method. Elbow Method tells the optimal cluster number for
        optimal inertia. Here this method gives 4 clusters for most optimal clustering, but we choose add 2 more
        clusters to get more rich partitioning.
        :param feature: name of dataframe column
        :param n_clusters: number of cluster
        :return: nothing
        """
        # build dataframe column
        df = pd.DataFrame()
        df[feature] = self.__data[feature]

        # fit KMeans
        model = KMeans(n_clusters=n_clusters)
        model.fit(df)
        pred = model.predict(df)

        # complete dataset with cluster label
        self.__data[feature + "_cluster"] = pred

    def __remake_prediction(self):
        """
        The size of clusters was determined by the Elbow method
        :return: None
        """
        cluster_conf = [
            {
                "feature": "NUM_BUY",
                "cluster_size": 4
            },
            {
                "feature": "SUM_PRICE",
                "cluster_size": 4
            },
            {
                "feature": "SIZE_BASKET",
                "cluster_size": 4
            }
        ]
        for roadmap in cluster_conf:
            self.__kmeans_prediction_one_feature(roadmap["feature"], roadmap["cluster_size"])

        # save prediction
        self.__data.to_pickle(segmentation_proc_cluster_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/segmentation_proc_cluster.pkl")

    def get_cluster_description(self, feature, cluster_label):
        tags = {
            "NUM_BUY": {
                0: "consommation normale",
                1: "consommation faible",
                2: "consommation forte",
                3: "consommation très forte"
            },
            "SUM_PRICE": {
                0: "budget petit",
                1: "budget moyen",
                2: "budget élevé",
                3: "budget très élevé"
            },
            "SIZE_BASKET": {
                0: "petit panier",
                1: "gros panier",
                2: "très gros panier",
                3: "moyen panier"
            }
        }
        target = {
            "feature": feature,
            "cluster": cluster_label,
            "tag": tags[feature][cluster_label],
            "tot_num_customers": self.__data_size
        }
        c = feature + "_cluster"
        cluster: pd.DataFrame = self.__data[self.__data[c] == cluster_label]
        target["cluster_size"] = len(cluster)
        target["cluster_proportion"] = round(target["cluster_size"] * 100.0 / self.__data_size, 2)
        target["min"] = int(cluster[feature].min())
        target["max"] = int(cluster[feature].max())
        target["mean"] = round(cluster[feature].mean(), 2)
        return target

    @staticmethod
    def __display_description(desc):
        feature = desc["feature"]
        percent = desc["cluster_proportion"]
        minn = desc["min"]
        maxx = desc["max"]
        moy = desc["mean"]
        str_return = ''
        if feature == "NUM_BUY":
            str_return += "\nNOMBRE D'ACHAT:\n"
            str_return += f"Ce client appartient au {percent} % de ceux qui achètent entre {minn} et {maxx}, avec une moyenne de {moy} achats."
        elif feature == "SUM_PRICE":
            str_return += "\nTOTAL DEPENSE:\n"
            str_return += f"Ce client appartient au {percent} % de ceux qui achètent des produits qui coûtent entre {minn} et {maxx} euros, avec un prix moyen d'un produit acheté de {moy} euros."
        elif feature == "SIZE_BASKET":
            str_return += "\nTAILLE MOYEN DU PANIER\n"
            str_return += f"Ce client appartient au {percent}% des clients qui font leur course avec des paniers de taille comprise entre {minn} et {maxx}, avec une taille moyenne de panier de {moy}."
        return str_return
    def get_description(self, user_id):
        all_desc = []
        for ft in ["NUM_BUY", "SUM_PRICE", "SIZE_BASKET"]:
            cluster_label = self.__data[self.__data["CLI_ID"] == user_id][ft + "_cluster"].iloc[0]
            desc = self.get_cluster_description(ft, cluster_label)
            all_desc.append(desc)
        tags = []
        for d in all_desc:
            tags.append(d["tag"])
        print("TAGS: " + ", ".join(tags))
        str_return = ''
        for d in all_desc:
            str_return += self.__display_description(d)
        return str_return