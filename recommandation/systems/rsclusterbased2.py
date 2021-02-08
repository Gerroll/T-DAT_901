import numpy as np
import pandas as pd
import json
from datetime import datetime
import hdbscan
from pathlib import Path
from inflect import engine
ie = engine()

"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent.parent
# path to processed data
processed_dir = project_dir.joinpath("assets").joinpath("processed")
clusterbased2_proc_file = processed_dir.joinpath('clusterbased2_proc.pkl')
clusterbased2_clust_file = processed_dir.joinpath('clusterbased2_clust.pkl')
# path to data source
kado_file = project_dir.joinpath("data").joinpath("KaDo.csv")


class Processor:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        # At the end of first processing, data processed dataframe are saved into pickle file
        if clusterbased2_proc_file.is_file():
            self.__data = pd.read_pickle(clusterbased2_proc_file)
        else:
            self.__process()

    def get_raw_data(self):
        return self.__raw_df

    def get_processed_data(self):
        return self.__data

    def __process(self):
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
        self.__data = pd.DataFrame(
            np.array(npa),
            columns=df_col
        )
        print(f"End preprocess at {datetime.now().time()}")

        # Save to pickle
        self.__data.to_pickle(clusterbased2_proc_file)
        print("File successfully saved to <PROJECT_ROOT>/assets/processed/clusterbased2_proc.pkl")


class RSClusterBased2:
    def __init__(self, remake_prediction=False, min_cluster_size=100):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None

        if remake_prediction:
            self.__remake_prediction(min_cluster_size)

        if not clusterbased2_clust_file.is_file():
            print("Predicted data doesn't exist. Re-compute the prediction...")
            self.__remake_prediction(min_cluster_size)

        self.__data = pd.read_pickle(clusterbased2_clust_file)

    def get_raw_df(self):
        return self.__raw_df

    def __remake_prediction(self, min_cluster_size):
        proc: Processor = Processor()
        self.__data = proc.get_processed_data()
        self.__train(min_cluster_size)

    def __train(self, min_cluster_size, save=True):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        df = pd.DataFrame()

        for fam in list(self.__raw_df["FAMILLE"].unique()):
            df[fam] = self.__data[fam]
        print("Start of fitting hdbscan model")
        clusterer.fit(df)
        print("End of fitting")
        print(f"Number of labels: {clusterer.labels_.max()}")
        self.__data["prediction"] = clusterer.labels_
        if save:
            self.__data.to_pickle(clusterbased2_clust_file)
            print("Predicted data successfully saved to <PROJECT_ROOT>/assets/processed/clusterbased2_clust.pkl")

    def __get_customers_id_from_cluster_id(self, cluster_id):
        """
        filter the processed data by the number of the cluster given, then retrieve index client corresponding
        :param cluster_id:
        :return: list of id
        """
        return list(self.__data[self.__data["prediction"] == cluster_id]["CLI_ID"])

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

    def __proportion_cluster_buy_it(self, df, df_size, libelle):
        tot_libelle = df[df["LIBELLE"] == libelle]["CLI_ID"].unique()
        return round(len(tot_libelle) * 100 / df_size, 3)

    def __compute_explanation(self, data):
        current_proportion = self.__proportion_to_string(data["current_customer"]["proportions"])
        cluster_proportion = self.__proportion_to_string(data["cluster_customer"]["proportions"])
        user_id = data["current_customer"]["ID"]
        cluster_size = len(data["cluster_customer"]["IDs"])
        df_raw_cluster = self.__raw_df[self.__raw_df["CLI_ID"].isin(data["cluster_customer"]["IDs"])]
        for i in range(len(data["recommendations"])):
            rank = ie.number_to_words(ie.ordinal(i+1))
            libelle = data["recommendations"][i]["LIBELLE"]
            prop = self.__proportion_cluster_buy_it(df_raw_cluster, cluster_size, libelle)
            explanation = f"{libelle} is the {rank} best recommendation for the customer {user_id}, because {prop}%" \
                          f" of customers of the same cluster bought it. Consumption characteristics user/cluster:" \
                          f" *User consumption type: {current_proportion} ; *Cluster consumption " \
                          f"type: {cluster_proportion} ."
            data["recommendations"][i]["explanation"] = explanation

    def get_recommendation(self, user_id):
        """
        Compute the n most recommended product (less if there is less)
        :param user_id: id of the user target of the recommendation
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
        self.__compute_explanation(data)
        return data["recommendations"]

    @staticmethod
    def __proportion_to_string(prop):
        s = []
        for key, val in prop.items():
            if val != "0.0%":
                s.append(f"{key}: {val}")
        return ", ".join(s)

    @staticmethod
    def __count_df_to_json(count):
        return [{"LIBELLE": x, "occurrence": y} for x, y in count.items()]


if __name__ == "__main__":
    rs: RSClusterBased2 = RSClusterBased2()

    prediction = rs.get_recommendation(996899213)
    print(json.dumps(prediction, indent=2))
