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

    def get_raw_df(self):
        return self.__raw_df

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

    def __get_customers_id_from_cluster_id(self, cluster_id):
        """
        filter the processed data by the number of the cluster given, then retrieve index client corresponding
        :param cluster_id:
        :return: list of id
        """
        return list(self.__data[self.__data["prediction"] == cluster_id]["CLI_ID"])

    def __proportion_to_string(self, prop):
        s = []
        for key, val in prop.items():
            if val != "0.0%":
                s.append(f"{key}: {val}")
        return ", ".join(s)

    def __get_famille_proportion(self, customer_ids):
        famille = ['HYGIENE', 'SOINS DU VISAGE', 'PARFUMAGE', 'SOINS DU CORPS', 'MAQUILLAGE', 'CAPILLAIRES',
                   'SOLAIRES', 'MULTI FAMILLES', 'SANTE NATURELLE']
        result = {}

        filtered = self.__data[self.__data["CLI_ID"].isin(customer_ids)]
        tot = 0.0
        for col in famille:
            s = filtered[col].sum()
            tot = tot + s
            result[col] = s
        for col in famille:
            result[col] = str(round(result[col] * 100.0 / tot, 2)) + '%'
        return result

    def __get_purchases_from_customer_ids(self, ids):
        # from the client IDs, retrieve all the transactions made by those clients
        filtered_by_user_ids = self.__raw_df[self.__raw_df["CLI_ID"].isin(ids)]

        # get only the name of the purchased product, then count their value
        return filtered_by_user_ids["LIBELLE"].value_counts()

    def __get_purchases_unique_from_customer_id(self, user_id):
        filtered = self.__raw_df[self.__raw_df["CLI_ID"] == user_id]
        return set(filtered["LIBELLE"].unique())

    def __count_df_to_json(self, count):
        return [{"LIBELLE": x, "occurrence": y} for x, y in count.items()]

    def __compute_explanation(self, data, n=0):
        first = data["recommendations"][n]["LIBELLE"]
        user_id = data["current_customer"]["ID"]
        current_proportion = self.__proportion_to_string(data["current_customer"]["proportions"])
        cluster_proportion = self.__proportion_to_string(data["cluster_customer"]["proportions"])
        cluster_size = len(data["cluster_customer"]["IDs"])
        number_of_user = 0
        for other_id in data["cluster_customer"]["IDs"]:
            if first in self.__get_purchases_unique_from_customer_id(other_id):
                number_of_user = number_of_user + 1
        prop = round(number_of_user * 100.0 / cluster_size, 2)

        return f"{first} is the best recommendation for the customer {user_id}, because this is the number {n+1} of " \
               f"products selling in this cluster, {prop}% of them buy it. " \
               f"Furthermore the user and his cluster have the same type of consumption: *User " \
               f"consumption type: {current_proportion} ; *Cluster consumption type: {cluster_proportion} ."

    def get_recommendation(self, user_id, n=10):
        """
        Compute the n most recommended product (less if there is less)
        :param user_id: id of the user target of the recommendation
        :param n: size of the product list recommended
        :return: a list of most buy element, the first is the most recommended
        """
        # Get the cluster label of the user
        cluster_label = self.__data[self.__data["CLI_ID"] == user_id]["prediction"].iloc[0]

        cluster_ids = self.__get_customers_id_from_cluster_id(cluster_label)

        # Get a list of the most buy product (ordered and indexed by their number)
        other_purchases = self.__get_purchases_from_customer_ids(cluster_ids)

        # Get a list of product purchased by a customer
        user_purchases = self.__get_purchases_from_customer_ids([user_id])
        user_purchases_unique = self.__get_purchases_unique_from_customer_id(user_id)

        # remove products already purchased by the current customer
        other_purchases_without_current = other_purchases[~other_purchases.index.isin(user_purchases_unique)]

        data = {
            "current_customer": {
                "ID": user_id,
                "proportions": self.__get_famille_proportion([user_id]),
                "purchases": self.__count_df_to_json(user_purchases)
            },
            "cluster_customer": {
                "IDs": cluster_ids,
                "proportions": self.__get_famille_proportion(cluster_ids),
                "purchases": self.__count_df_to_json(other_purchases)
            },
            "recommendations": self.__count_df_to_json(other_purchases_without_current),
        }

        explanation = self.__compute_explanation(data)
        data["explanation"] = explanation

        return data


if __name__ == "__main__":
    rs: RSClusterBased = RSClusterBased()
    prediction = rs.get_recommendation(996899213)
    print(json.dumps(prediction, indent=2))
