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
segmentation2_proc_file = proc_data_dir.joinpath("segmentation2_proc.pkl")
segmentation2_proc_cluster_file = proc_data_dir.joinpath("segmentation2_proc_cluster.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")


class Processor:
    def __init__(self, raw_data):
        self.__raw_df = raw_data
        self.__data = None
        # At the end of first processing, data processed dataframe are saved into pickle file
        if segmentation2_proc_file.is_file():
            self.__load_file()
        else:
            self.__process()

    def get_raw_data(self):
        return self.__raw_df

    def get_processed_data(self):
        return self.__data

    def __load_file(self):
        self.__data = pd.read_pickle(segmentation2_proc_file)

    def __process(self):
        # Retrieve all unique client ID
        client_ids = self.__raw_df["CLI_ID"].unique()

        # build dataframe column
        df_col = ["CLI_ID", "JAN", "FEB", "MAR", "APR", "MAY", "JUI", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        # collect data
        print(f"Start collecting at {datetime.now().time()}")
        collect = {k: [0] * 13 for k in client_ids}
        for row in self.__raw_df.itertuples():
            cli_id = row[8]
            month = row[2]
            collect[cli_id][month - 1] += 1
            collect[cli_id][12] += 1

        def proportion(val):
            tot = val[12]
            return [round(val[i] / tot, 3) for i in range(0, 12)]

        # build result
        print(f"Start building dataframe at {datetime.now().time()}")
        npa = [[key] + proportion(val) for key, val in collect.items()]
        self.__data = pd.DataFrame(
            np.array(npa),
            columns=df_col
        )
        print(f"End preprocess at {datetime.now().time()}")

        # Save to pickle
        self.__data.to_pickle(segmentation2_proc_file)
        print("File successfully saved to <PATH_TO_PATH>/processed-data/segmentation2_proc.pkl")


class Clusterer2:
    def __init__(self, raw_data):
        proc = Processor(raw_data)
        self.__raw_df = proc.get_raw_data()
        self.__data = proc.get_processed_data()
        self.__data_size = len(self.__data)
        if segmentation2_proc_cluster_file.is_file():
            self.__load_predicted_data()
        else:
            self.__remake_prediction()

    def get_predicted_data(self):
        return self.__data

    def __load_predicted_data(self):
        self.__data = pd.read_pickle(segmentation2_proc_cluster_file)

    def __kmeans_prediction_one_feature(self, features, cluster_name, n_clusters):
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
        for ft in features:
            df[ft] = self.__data[ft]

        # fit KMeans
        model = KMeans(n_clusters=n_clusters)
        model.fit(df)
        pred = model.predict(df)

        # complete dataset with cluster label
        self.__data[cluster_name] = pred

    def __remake_prediction(self):
        """
        The size of clusters was determined by the Elbow method
        :return: None
        """
        cluster_conf = [
            {
                "cluster_name": "cluster",
                "feature": ["JAN", "FEB", "MAR", "APR", "MAY", "JUI", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
                "cluster_size": 13
            }
        ]
        for roadmap in cluster_conf:
            self.__kmeans_prediction_one_feature(roadmap["feature"], roadmap["cluster_name"], roadmap["cluster_size"])

        # save prediction
        self.__data.to_pickle(segmentation2_proc_cluster_file)
        print("File successfully saved to <PATH_TO_PATH>/processed-data/segmentation2_proc_cluster_3.pkl")

    def get_description(self, user_id):
        cluster_label = self.__data[self.__data["CLI_ID"] == user_id]["cluster"].iloc[0]
        cluster = self.__data[self.__data["cluster"] == cluster_label]
        return_str = ''
        del cluster["CLI_ID"]
        del cluster["cluster"]
        p = round(len(cluster) * 100.0 / self.__data_size, 2)
        if cluster_label == 2:
            return_str += f"Ce client appartient au {p} % de ceux qui achètent en proportion similaire tout au long de l'année."
        else:
            months = {
                "JAN": "janvier",
                "FEB": "février",
                "MAR": "mars",
                "APR": "avril",
                "MAY": "mai",
                "JUI": "juin",
                "JUL": "juillet",
                "AUG": "aout",
                "SEP": "septembre",
                "OCT": "octobre",
                "NOV": "novembre",
                "DEC": "decembre"
            }
            m = cluster.mean().idxmax()
            return_str += f"Ce client appartient au {p} % de ceux qui font majoritairement leurs achats au mois de {months[m]}"
        return return_str