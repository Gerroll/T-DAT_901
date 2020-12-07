import numpy as np
import pandas as pd
import json
from datetime import datetime
import hdbscan
from pathlib import Path

"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent
# path to processed data
proc_data_dir = project_dir.joinpath("processed-data")
user_proc_cluster_file = proc_data_dir.joinpath("user_proc_cluster.pkl")
user_proc_file = proc_data_dir.joinpath("user_proc.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")


class Processor:
    def __init__(self):
        self.__data = None
        if user_proc_file.is_file():
            self.__load_file()
        else:
            self.__raw_df = pd.read_csv(kado_file)

    def run(self, save=True):
        self.__data = self.__preprocess()
        if save:
            self.__data.to_pickle(user_proc_file)
            print("File successfully saved to <PATH_TO_PATH>/processed-data/user_proc.pkl")

    def get_data(self):
        """
        Get the dataframe result of the process
        :return: dataframe result
        """
        if self.__data is None:
            raise Exception('Data is not processed and worth null')
        return self.__data

    def __load_file(self):
        self.__data = pd.read_pickle(user_proc_file)

    def __preprocess(self):
        # Retrieve all unique client ID
        client_ids = self.__raw_df["CLI_ID"].unique()

        # build dataframe column
        famille_list = list(self.__raw_df["FAMILLE"].unique())
        df_col = ["CLI_ID"] + famille_list

        # collect data
        print(f"Start collecting at {datetime.now().time()}")
        collect = {k: len(famille_list) * [0] for k in client_ids}
        for row in self.__raw_df.itertuples():
            cli_id = row[8]
            famille_id = row[4]
            index_maille = famille_list.index(famille_id)
            collect[cli_id][index_maille] += 1

        # build result
        print(f"Start building dataframe at {datetime.now().time()}")
        npa = [[key] + value for key, value in collect.items()]
        result: pd.DataFrame = pd.DataFrame(
            np.array(npa),
            columns=df_col
        )

        print(f"End preprocess at {datetime.now().time()}")
        return result


class RSClusterBased:
    def __init__(self, remake_prediction=False, min_cluster_size=60):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        if remake_prediction:
            proc: Processor = Processor()
            self.__data = proc.get_data()
            self.__remake_prediction(min_cluster_size)
        else:
            self.__load_predicted_data()

    def __load_predicted_data(self):
        if not user_proc_cluster_file.is_file():
            raise Exception("Predicted data doesn't exist. Re-compute the prediction.")
        self.__data = pd.read_pickle(user_proc_cluster_file)

    def __remake_prediction(self, min_cluster_size, save=True):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        df = pd.DataFrame()

        for fam in list(self.__raw_df["FAMILLE"].unique()):
            df[fam] = self.__data[fam]
        clusterer.fit(df)
        print(clusterer.labels_.max())
        self.__data["prediction"] = clusterer.labels_
        if save:
            self.__data.to_pickle(user_proc_cluster_file)
            print("File successfully saved to <PATH_TO_PATH>/processed-data/user_proc_cluster.pkl")

    def __get_most_buy_from_cluster_id(self, cluster_id):
        # filter the processed data by the number of the cluster given, then retrieve index client corresponding
        cluster_user_ids = list(self.__data[self.__data["prediction"] == cluster_id]["CLI_ID"])

        # from the client index of the cluster, retrieve all the transactions made by those clients
        filtered_by_user_ids = self.__raw_df[self.__raw_df["CLI_ID"].isin(cluster_user_ids)]

        # get only the name of the purchased product, then count their value
        return filtered_by_user_ids["LIBELLE"].value_counts()

    def __get_prod_buy_from_user_id(self, user_id):
        filtered = self.__raw_df[self.__raw_df["CLI_ID"] == user_id]
        return list(filtered["LIBELLE"].unique())

    def get_recommendation(self, user_id, n=10):
        """
        Compute the n most recommended product (less if there is less)
        :param user_id: id of the user target of the recommendation
        :param n: size of the product list recommended
        :return: a list of most buy element, the first is the most recommended
        """
        # Get the cluster number of the user
        cluster_id = self.__data[self.__data["CLI_ID"] == user_id]["prediction"].iloc[0]

        # Get a list of the most buy product (ordered and indexed by their number)
        most_buy_cluster = self.__get_most_buy_from_cluster_id(cluster_id)
        print(f"Size of most buy products from cluster: {len(most_buy_cluster)}")

        # Get a list of product purchased by a customer
        from_user = self.__get_prod_buy_from_user_id(user_id)

        i = 0
        target = []
        for index, values in most_buy_cluster.items():
            if index not in from_user:
                target.append(index)
            if i >= n:
                break
            i = i + 1
        return target


if __name__ == "__main__":
    rs: RSClusterBased = RSClusterBased()
    prediction = rs.get_recommendation(1490281)
    print(prediction)
