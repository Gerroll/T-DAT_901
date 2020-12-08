import numpy as np
import pandas as pd
import pathlib
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from datetime import datetime


"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent
# path to processed data
proc_data_dir = project_dir.joinpath("processed-data")
user_proc_file = proc_data_dir.joinpath("user_proc_2.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")
# label mapping
label_mapping_file = project_dir.joinpath("label_mapping.json")


class Processor:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        if user_proc_file.is_file():
            self.__load_file()
        # keep original label from product
        self.__is_collect_label = False
        self.label_mapping = {}

    def run(self, save=True):
        self.__preprocess()
        if save:
            self.__data.to_pickle(user_proc_file)
            print("File successfully saved to <PATH_TO_PATH>/processed-data/user_proc_2.pkl")

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

        # collect data
        print(f"Start collecting at {datetime.now().time()}")
        collect = {k: set() for k in client_ids}
        for row in self.__raw_df.itertuples():
            cli_id = row[8]
            item_id = row[7]
            conv_item = item_id.replace(" ", "").replace(".", "").replace("/", "").lower()
            collect[cli_id].add(conv_item)

        # build result
        print(f"Start building at {datetime.now().time()}")
        # npa = [[key, ' '.join(value)] for key, value in collect.items()]
        r = {"CLI_ID": [], "description": []}
        for key, value in collect.items():
            r["CLI_ID"].append(key)
            r["description"].append(' '.join(value))
        result: pd.DataFrame = pd.DataFrame(r)

        print(f"End preprocess at {datetime.now().time()}")
        self.__data = result


class Counter:
    def __init__(self):
        self.__count = {}
        self.__size = 0

    def get_count(self):
        return self.__count

    def len(self):
        return self.__size

    def add(self, item: str):
        self.__size = self.__size + 1
        if item not in self.__count:
            self.__count[item] = 0
        self.__count[item] = self.__count[item] + 1

    def extend(self, collection):
        for item in collection:
            self.add(item)


class RSUserBased:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = Processor().get_data()

    def __index_to_id(self, index):
        return self.__data.loc[index]["CLI_ID"]

    def __index_to_id_from_list(self, indexes):
        target = []

        for i in indexes:
            target.append(self.__data.loc[i]["CLI_ID"])
        return target

    def __is_same_user(self, id_1, id_2):
        """
        Here, users are consider the same if they are buy the same products
        :param id_1: id of user 1
        :param id_2: id of user 2
        :return: an empty list if the users are the same, if not, a list that contains the product difference.
        """
        items_1 = self.__get_prod_buy_from_user_id(id_1)
        items_2 = self.__get_prod_buy_from_user_id(id_2)
        return items_1.symmetric_difference(items_2)

    def __get_most_buy_from_user_ids(self, ids):
        filtered = self.__raw_df[self.__raw_df["CLI_ID"].isin(ids)]
        return filtered["LIBELLE"].value_counts()

    def __get_prod_buy_from_user_id(self, user_id):
        filtered = self.__raw_df[self.__raw_df["CLI_ID"] == user_id]
        return set(filtered["LIBELLE"].unique())

    def get_recommendation(self, user_id, n=10):
        """
        Compute a recommendation from user description dataset using CountVectorize model from scikit-learn library
        :param user_id: id of the user target of the recommendation
        :param n: number of similar user's product used to make the recommendation
        :return: a list of production recommended
        """
        # Retrieve the description associated with user id
        user_description = self.__data[self.__data["CLI_ID"] == user_id]["description"].iloc[0]

        # Collection of text converter into a matrix of token counts
        count = CountVectorizer()
        count.fit(self.__data["description"])
        count_matrix = count.transform(self.__data["description"])

        # Transform this description into vector
        user_vector = count.transform([user_description])
        # Compute similarity vector
        [similarity] = cosine_similarity(user_vector, count_matrix)

        sim_scores = list(enumerate(similarity))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # The first most similar user is necessarily himself, so we remove it from the list
        sim_scores.pop(0)

        # Collect products to recommend
        i = 0
        target: Counter = Counter()
        while target.len() < n:
            closer_index, closer_val = sim_scores[i]
            i = i + 1
            closer_id = self.__index_to_id(closer_index)
            difference = self.__is_same_user(closer_id, user_id)
            target.extend(difference)

        # formatting result and return
        return [x for x, y in sorted(target.get_count().items(), key=lambda x: x[1], reverse=True)]


if __name__ == "__main__":
    #proc: Processor = Processor()
    #proc.run()
    rs: RSUserBased = RSUserBased()
    prediction = rs.get_recommendation(1490281)
    # ordered by the most recommend to the less
    print(prediction)
