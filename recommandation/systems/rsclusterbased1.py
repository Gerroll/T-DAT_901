import pandas as pd
from sklearn.cluster import KMeans
from enum import Enum
from pathlib import Path
from datetime import datetime
from json import dumps
from inflect import engine
ie = engine()

"""
    Pandas option
"""
pd.set_option("display.max_columns", None)

"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent.parent
# path to processed data
processed_dir = project_dir.joinpath('assets').joinpath('processed')
clusterbased1_clust_file = processed_dir.joinpath('clusterbased1_clust.pkl')
# path to data source
kado_file = project_dir.joinpath("data").joinpath("KaDo.csv")


class Category(Enum):
    FAMILLE = "FAMILLE"
    MAILLE = "MAILLE"
    UNIVERS = "UNIVERS"


class Column(Enum):
    TICKET_ID = "TICKET_ID"
    MOIS_VENTE = "MOIS_VENTE"
    PRIX_NET = "PRIX_NET"
    FAMILLE = "FAMILLE"
    UNIVERS = "UNIVERS"
    MAILLE = "MAILLE"
    LIBELLE = "LIBELLE"
    CLI_ID = "CLI_ID"


class RSClusterBased1:
    def __init__(self):
        self.__raw_df = pd.read_csv(kado_file)
        self.__data = None
        self.famille_list = self.__raw_df["FAMILLE"].unique()

        if clusterbased1_clust_file.is_file():
            self.__data = pd.read_pickle(clusterbased1_clust_file)
        else:
            print("Predicted data doesn't exist. Re-compute the prediction...")
            self.__remake_prediction()

    #
    # RECOMANDATION BY SCORE AND FAMILLY PREFERENCE
    #
    def most_popular_famille(self, data):
        """For a given Dataset, the most popular items with their libelle and their count  """
        FAMILLEUNIVERS = data.groupby(['FAMILLE', 'LIBELLE']).size().to_frame(name='size').reset_index().sort_values(
            by=['size'], ascending=False)
        return FAMILLEUNIVERS.drop_duplicates(subset=['FAMILLE'])

    # Score definition: Number of client that buy the article at least twice
    def get_score_for_libelle_df(self, metadata):
        # Table of all "CLI_ID", "LIBELLE" possible then counting the "NB_BUY" for each
        # table columns : CLI_ID, LIBELLE, NB_BUY
        cliId_libelle = metadata[["CLI_ID", "LIBELLE"]].copy().groupby(["CLI_ID", "LIBELLE"]).size().to_frame(
            name='NB_BUY').sort_values(by=['NB_BUY'])

        # decrement size colum to see what item was buy twice
        cliId_libelle['NB_BUY'] -= 1

        # replace nb of buyed an item by client to 1
        cliId_libelle["NB_BUY"].mask(cliId_libelle["NB_BUY"] >= 1, 1)

        # count all client that bought at least twice the product
        # table columns: LIBELLE, SCORE
        return cliId_libelle.groupby(["LIBELLE"]).size().to_frame(name='SCORE').sort_values(by=['SCORE'], ascending=False).reset_index()

    #
    #  return a dataFrame
    #  table columns: FAMILLE, MAILLE, UNIVERS, LIBELLE, SCORE
    #
    def get_libelle_score_df_with_categories(self, metadata):
        libelleScore = self.get_score_for_libelle_df(metadata).sort_values(by=['LIBELLE'])
        universLibelle = metadata[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']].copy().groupby(
            ['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index().sort_values(by=['LIBELLE'])
        return pd.merge(universLibelle[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']], libelleScore[['LIBELLE', 'SCORE']],
                        on=['LIBELLE'], how='outer').sort_values(by=['SCORE'], ascending=False).reset_index()[
            ['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE', 'SCORE']]

    # return a list of string that represent buying item of one client
    def get_list_of_libelle_of_the_client_did_buy(self, metadata, client_id: int):
        df = metadata.copy()
        client_list_article_dataframe = df.loc[df[Column.CLI_ID.name] == client_id]
        label_list_df = client_list_article_dataframe.groupby(
            ['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index()
        return label_list_df

    def get_not_buy_label_df_with_score_df(self, metadata, client_id):
        libelleScoreDFWithCategories = self.get_libelle_score_df_with_categories(metadata)
        list_buyed_item_client = self.get_list_of_libelle_of_the_client_did_buy(metadata, client_id)['LIBELLE'].tolist()
        df_filter_score_label = libelleScoreDFWithCategories.loc[
            ~libelleScoreDFWithCategories['LIBELLE'].isin(list_buyed_item_client)].reset_index()
        return df_filter_score_label

    def get_five_first_line_libelle_of_specific_category(self, libelleScoreDFWithCategories, category: Category,
                                                         nameCategory: str):
        return libelleScoreDFWithCategories.loc[libelleScoreDFWithCategories[category.name] == nameCategory]

    def get_recommendation_strategy(self, metadata, client_id, famille_preferred):
        preferred_famille_of_client = famille_preferred
        df_client_unbuyed_label_with_score = self.get_not_buy_label_df_with_score_df(metadata, client_id)
        return self.get_five_first_line_libelle_of_specific_category(df_client_unbuyed_label_with_score,
                                                                         Category.FAMILLE, preferred_famille_of_client)

    #
    # RECOMMENDATION BY SCORE AND FAMILY PREFERENCE USING CLUSTERING BEFORE SCORING
    #
    def get_user_proportion_by_family(self, metadata, famille_list):
        df = metadata.groupby(["CLI_ID", "FAMILLE"]).size().to_frame(name='NB').reset_index()
        userPercentByFamilly = []
        CLI_ID = index = -1
        default_famille = {}
        for f in famille_list:
            default_famille[f] = 0
        for _, row in df.iterrows():  # "row" variable is like
            if CLI_ID != row["CLI_ID"]:  # CLI_ID     1490281
                index += 1  # FAMILLE    HYGIENE
                CLI_ID = row["CLI_ID"]  # NB               3
                userPercentByFamilly.append({  # Name: 0, dtype: object
                    'CLI_ID': CLI_ID,
                    'FAMILLE': default_famille.copy()
                })
            userPercentByFamilly[index]['FAMILLE'][row['FAMILLE']] = row['NB']
        return userPercentByFamilly

    def user_proportion_by_famille_to_kmean_format(self, userPercentByFamilly, famille_list):
        data_k_mean = {'CLI_ID': []}
        for f in famille_list:
            data_k_mean[f] = []
        for u in userPercentByFamilly:
            itemBuyed = 0
            for famille, value in u['FAMILLE'].items():
                itemBuyed += value
            for famille, value in u['FAMILLE'].items():
                val = int(value * 100 / itemBuyed)
                data_k_mean[famille].append(val)
            data_k_mean['CLI_ID'].append(u['CLI_ID'])
        return data_k_mean

    def get_cli_id_list_of_cli_cluster_group(self, predictedDf, cli_id):
        cluster_label = predictedDf[predictedDf['CLI_ID'] == cli_id]['cluster'].iloc[0]
        return list(predictedDf[predictedDf['cluster'] == cluster_label]['CLI_ID'])

    def __remake_prediction(self):
        # getting a model that register all user with their CLID_ID and their percent of buyed item by familly
        # [{
        #     'CLI_ID': 123456,
        #     'FAMILLE': {
        #         'HYGIENE': 10,
        #         . . .
        #         'SOINS DU CORPS': 10,
        #     }
        # },
        # {
        #     . . .
        # }]
        userPercentByFamilly = self.get_user_proportion_by_family(self.__raw_df, self.famille_list)

        # build dataframe from model
        kmeanFormat = self.user_proportion_by_famille_to_kmean_format(userPercentByFamilly, self.famille_list)
        self.__data = pd.DataFrame(kmeanFormat)

        # keep only famille column (without CLI_ID)
        familyProportionDf = pd.DataFrame()
        for f in self.famille_list:
            familyProportionDf[f] = self.__data[f]

        # fit kmeans
        kmeans = KMeans(n_clusters=8, random_state=1).fit(familyProportionDf)

        # store prediction into dataframe
        self.__data['cluster'] = kmeans.labels_

        # save prediction to pickle
        self.__data.to_pickle(clusterbased1_clust_file)
        print("Predicted data successfully saved to <PROJECT_ROOT>/assets/processed/clusterbased1_clust.pkl")

    def recommendation_to_json(self, df):
        return [{"LIBELLE": row[5], "SCORE": row[6]} for row in df.itertuples()]

    def compute_prop_by_family_and_user(self, user_id, family):
        tot = self.__raw_df[self.__raw_df["CLI_ID"] == user_id]
        tot_family = tot[tot["FAMILLE"] == family]
        return round(len(tot_family) * 100 / len(tot), 2)

    def compute_prop_by_cluster(self, df, l_df, libelle):
        tot_libelle = df[df["LIBELLE"] == libelle]["CLI_ID"].unique()
        return round(len(tot_libelle) * 100 / l_df, 3)

    def complete_data(self, data):
        user_id = data["user_id"]
        cluster_ids = data["cluster_ids"]
        preferred_family = data["preferred_family"]
        prop_family = data["proportion_family"]
        filtered = self.__raw_df[(self.__raw_df["FAMILLE"] == preferred_family) & (self.__raw_df["CLI_ID"].isin(cluster_ids))]
        tot = len(filtered["CLI_ID"].unique())

        for i in range(len(data["recommendations"])):
            libelle = data["recommendations"][i]["LIBELLE"]
            rank = ie.number_to_words(ie.ordinal(i+1))
            prop_family_cluster = self.compute_prop_by_cluster(filtered, tot, libelle)
            explanation = f"{libelle} is the {rank} best recommendation for the customer {user_id}, because the preferred" \
                          f" family is {preferred_family} ({prop_family}% of its purchases). And, for this family," \
                          f" {prop_family_cluster}% of his cluster bought it."
            data["recommendations"][i]["explanation"] = explanation

    def get_recommendation(self, user_id):
        # retrieve the list client id from the cluster of the current user id
        cliIdList = self.get_cli_id_list_of_cli_cluster_group(self.__data, user_id)

        # getting a dataframe with only ticket with client_id in the list
        clientClusterMetadata = self.__raw_df.loc[self.__raw_df['CLI_ID'].isin(cliIdList)].sort_values(
            by=['CLI_ID']).reset_index()

        # getting preferred family for my_client_id
        clientMostPopularFamilleDf = self.most_popular_famille(self.__raw_df.loc[self.__raw_df['CLI_ID'].isin([user_id])])
        clientMostPopularFamille = clientMostPopularFamilleDf["FAMILLE"].tolist()[0]

        # getting recommendation item based on best score and preferred family of my client
        recommendation = self.get_recommendation_strategy(clientClusterMetadata, user_id, clientMostPopularFamille)
        prop_family = self.compute_prop_by_family_and_user(user_id, clientMostPopularFamille)
        data = {
            "user_id": user_id,
            "preferred_family": clientMostPopularFamille,
            "recommendations": self.recommendation_to_json(recommendation),
            "proportion_family": prop_family,
            "cluster_ids": cliIdList
        }
        self.complete_data(data)
        return data["recommendations"]


if __name__ == "__main__":
    clusterer = RSClusterBased1()
    print(dumps(clusterer.get_recommendation(996899213), indent=4))
    #print(clusterer.get_recommendation(1490281))

