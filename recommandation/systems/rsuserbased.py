import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from datetime import datetime
from json import dumps


"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent.parent
# path to processed data
proc_data_dir = project_dir.joinpath("processed-data")
user_proc_file = proc_data_dir.joinpath("user_proc_2.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")


class Processor:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        # At the end of first processing, data processed dataframe are saved into pickle file
        if user_proc_file.is_file():
            self.__load_file()
        else:
            self.__process()

    def get_raw_data(self):
        return self.__raw_df

    def get_processed_data(self):
        return self.__data

    def __load_file(self):
        self.__data = pd.read_pickle(user_proc_file)

    def __process(self):
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
        json_result = {"CLI_ID": [], "description": []}
        for key, value in collect.items():
            json_result["CLI_ID"].append(key)
            json_result["description"].append(' '.join(value))

        self.__data = pd.DataFrame(json_result)
        print(f"End preprocess at {datetime.now().time()}")

        # Save dataframe into pickle file
        self.__data.to_pickle(user_proc_file)
        print("File successfully saved to <PROJECT_ROOT>/processed-data/user_proc_2.pkl")


class Counter:
    def __init__(self):
        self.__count = {}

    def get_count(self):
        return self.__count

    def len(self):
        return len(self.__count)

    def max_occur(self):
        return max(self.__count.values())

    def add(self, item: str):
        if item not in self.__count:
            self.__count[item] = 0
        self.__count[item] = self.__count[item] + 1

    def extend(self, collection):
        for item in collection:
            self.add(item)


class RSUserBased:
    def __init__(self):
        processor = Processor()
        self.__raw_df = processor.get_raw_data()
        self.__data = processor.get_processed_data()

    def __index_to_cli_id(self, index):
        return self.__data.loc[index]["CLI_ID"]

    def __get_unique_prod_buy_from_user_id(self, user_id):
        filtered = self.__raw_df[self.__raw_df["CLI_ID"] == user_id]
        return set(filtered["LIBELLE"].unique())

    def __complete_dict_customer_purchase(self, related_customers, customer_id, sim_val=1.0):
        related_customers[str(customer_id)] = {
            "sim_value": sim_val,
            "purchases": {}
        }
        # get items buying by the customers
        customer_transaction: pd.DataFrame = self.__raw_df[self.__raw_df["CLI_ID"] == customer_id]
        for item, count in customer_transaction["LIBELLE"].value_counts().items():
            related_customers[str(customer_id)]["purchases"][item] = count

    def __compute_description(self, data):
        first = data["recommendations"][0]["LIBELLE"]
        user_id = list(data["current_customer"].keys())[0]
        number_of_user = 0
        sim_average = 0
        for key, val in data["related_customers"].items():
            purchases = list(val["purchases"].keys())
            if first in purchases:
                number_of_user = number_of_user + 1
                sim_average = sim_average + val["sim_value"]
        sim_average = round(sim_average / number_of_user, 3)

        return f"{first} is the best recommendation for the customer {user_id}, because {number_of_user} others," \
            f" who have on average a similarity of {sim_average}, buy the same product."

    def get_recommendation(self, user_id):
        """
        Compute a recommendation from user description dataset using CountVectorize model from scikit-learn library.
        A arbitrary choice is too make recommendation with all users whose get 50% of similarity with the current
        user, so the recommendation is an aggregation of their consumption.
        :param user_id: id of the user target of the recommendation
        :return: a dictionary with the current user id and his purchases, the customers, their score of similarity with
        the current user and their purchases and the list of recommendation with reversed order
        """
        # Retrieve the description associated with user id
        user_description = self.__data[self.__data["CLI_ID"] == user_id]["description"].iloc[0]

        # Will contains current customer and his purchases
        current_customer = {}
        # Will contains other related customers and their purchases
        related_customers = {}

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

        # Collect product from current costumer
        self.__complete_dict_customer_purchase(current_customer, user_id)

        # Collect products to recommend
        i = 0
        closer_index, closer_val = sim_scores[i]
        user_items = self.__get_unique_prod_buy_from_user_id(user_id)
        target: Counter = Counter()
        while closer_val >= 0.5:
            closer_id = self.__index_to_cli_id(closer_index)
            # Get products in closer_items but not in user_items regardless amount.
            closer_items = self.__get_unique_prod_buy_from_user_id(closer_id)
            difference = closer_items - user_items
            if len(difference) != 0:
                self.__complete_dict_customer_purchase(related_customers, closer_id, closer_val)
                target.extend(difference)
            i = i + 1
            closer_index, closer_val = sim_scores[i]

        # formatting result and return
        data = {
            "current_customer": current_customer,
            "related_customers": related_customers,
            "recommendations": [{"LIBELLE": x, "occurrence": y} for x, y in sorted(target.get_count().items(), key=lambda x: x[1], reverse=True)]
        }

        description = self.__compute_description(data)
        data["description"] = description

        return data


if __name__ == "__main__":
    rs: RSUserBased = RSUserBased()
    prediction = rs.get_recommendation(996899213)
    # ordered by the most recommend to the less
    print(dumps(prediction, indent=2))
