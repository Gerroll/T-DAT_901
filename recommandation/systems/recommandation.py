import random
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from enum import Enum
from pathlib import Path

"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent.parent
# path to processed data
proc_data_dir = project_dir.joinpath("processed-data")
user_proc_cluster_file = proc_data_dir.joinpath("user_proc_cluster.pkl")
user_proc_file = proc_data_dir.joinpath("user_proc.pkl")
# path to data source
data_dir = project_dir.joinpath("data")
kado_file = data_dir.joinpath("KaDo.csv")


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


def init_dataframe(nrows=100000):
    metadata = pd.read_csv(kado_file, low_memory=False, nrows=nrows)
    return metadata


# just for the sick of it
def print_basic_data(metadata):
    print(metadata.head(1))
    print()
    print(len(metadata["LIBELLE"].value_counts(dropna=False)))
    print()
    print(len(metadata["CLI_ID"].value_counts(dropna=False)))
    print()
    print(len(metadata["UNIVERS"].value_counts(dropna=False)))
    print()
    print(len(metadata["MAILLE"].value_counts(dropna=False)))
    print()
    print(len(metadata["FAMILLE"].value_counts(dropna=False)))


############
###
### RECOMANDATION BY SCORE AND FAMILLY PREFERENCE
###
############
def most_popular_famille(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    FAMILLEUNIVERS = data.groupby(['FAMILLE','LIBELLE']).size().to_frame(name='size').reset_index().sort_values(by=['size'],ascending=False)
    return FAMILLEUNIVERS.drop_duplicates(subset=['FAMILLE'])
     

# Score definition: Number of client that buy the article at least twice
def get_score_for_libelle_df(metadata):
    # Table of all "CLI_ID", "LIBELLE" possible then counting the "NB_BUY" for each
    # table columns : CLI_ID, LIBELLE, NB_BUY
    cliId_libelle =  metadata[["CLI_ID", "LIBELLE"]].copy().groupby(["CLI_ID", "LIBELLE"]).size().to_frame(name = 'NB_BUY').sort_values(by=['NB_BUY'])

    # decrement size colum to see what item was buy twice
    cliId_libelle['NB_BUY'] -= 1

    # replace nb of buyed an item by client to 1
    cliId_libelle["NB_BUY"].mask(cliId_libelle["NB_BUY"] >= 1, 1)

    # count all client that bought at least twice the product
    # table columns: LIBELLE, SCORE
    return cliId_libelle.groupby(["LIBELLE"]).size().to_frame(name = 'SCORE').sort_values(by=['SCORE'], ascending=False).reset_index()


#
#  return a dataFrame
#  table columns: FAMILLE, MAILLE, UNIVERS, LIBELLE, SCORE
#
def get_libelle_score_df_with_categories(metadata):
    libelleScore = get_score_for_libelle_df(metadata).sort_values(by=['LIBELLE'])
    universLibelle = metadata[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']].copy().groupby(['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index().sort_values(by=['LIBELLE'])
    return pd.merge(universLibelle[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']], libelleScore[['LIBELLE', 'SCORE']], on=['LIBELLE'], how='outer').sort_values(by=['SCORE'], ascending=False).reset_index()[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE', 'SCORE']]


# return a list of string that represent buying item of one client
def get_list_of_libelle_of_the_client_did_buy(metadata, client_id: int):
    df = metadata.copy()
    client_list_article_dataframe = df.loc[df[Column.CLI_ID.name] == client_id]
    label_list_df = client_list_article_dataframe.groupby(['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index()
    return label_list_df


def get_list_of_libelle_of_the_client_did_not_buy(metadata, client_id: int):
    df = get_libelle_score_df_with_categories(metadata)
    list_buyed_label = get_list_of_libelle_of_the_client_did_buy(metadata, client_id)['LIBELLE'].tolist()
    df_by_label = df.groupby(['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index()
    df_label_never_buyed = df_by_label.loc[~df_by_label['LIBELLE'].isin(list_buyed_label)].reset_index()[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']]
    return df_label_never_buyed


def get_not_buy_label_df_with_score_df(metadata, client_id):
    libelleScoreDFWithCategories = get_libelle_score_df_with_categories(metadata)
    list_buyed_item_client = get_list_of_libelle_of_the_client_did_buy(metadata, client_id)['LIBELLE'].tolist()
    df_filter_score_label = libelleScoreDFWithCategories.loc[~libelleScoreDFWithCategories['LIBELLE'].isin(list_buyed_item_client)].reset_index()
    return df_filter_score_label


def get_five_first_line_libelle_of_specific_category(libelleScoreDFWithCategories, category: Category, nameCategory: str):
    return libelleScoreDFWithCategories.loc[libelleScoreDFWithCategories[category.name] == nameCategory][:5]


def get_recommendation_strategie_1(metadata, client_id, famille_prefered):
    prefered_famille_of_client = famille_prefered
    df_client_unbuyed_label_with_score = get_not_buy_label_df_with_score_df(metadata, client_id)
    recomandation = get_five_first_line_libelle_of_specific_category(df_client_unbuyed_label_with_score, Category.FAMILLE, prefered_famille_of_client)
    return recomandation


#
# RECOMMENDATION BY SCORE AND FAMILY PREFERENCE USING CLUSTERING BEFORE SCORING
#
def get_user_proportion_by_family(metadata, famille_list):
    df = metadata.groupby(["CLI_ID", "FAMILLE"]).size().to_frame(name = 'NB').reset_index()
    userPercentByFamilly = []
    CLI_ID = index = -1
    default_famille = {}
    for f in famille_list:
        default_famille[f] = 0
    for _, row in df.iterrows():                # "row" variable is like
        if CLI_ID != row["CLI_ID"]:             # CLI_ID     1490281
            index += 1                          # FAMILLE    HYGIENE
            CLI_ID = row["CLI_ID"]              # NB               3
            userPercentByFamilly.append({       # Name: 0, dtype: object
                'CLI_ID': CLI_ID,
                'FAMILLE': default_famille.copy()
            })
        userPercentByFamilly[index]['FAMILLE'][row['FAMILLE']] = row['NB']
    return userPercentByFamilly


def user_proportion_by_famille_to_kmean_format(userPercentByFamilly, famille_list):
    data_k_mean = {}
    for f in famille_list:
        data_k_mean[f] = []
    for u in userPercentByFamilly:
        itemBuyed = 0
        for famille, value in u['FAMILLE'].items():
            itemBuyed += value
        for famille, value in u['FAMILLE'].items():
            val = int(value * 100 / itemBuyed)
            data_k_mean[famille].append(val)
    return data_k_mean


def get_cli_index_from_cli_id(userPercentByFamilly, client_id):
    for i, user in enumerate(userPercentByFamilly):
        if user['CLI_ID'] == client_id:
            return i
    return 0


def create_client_for_kmean_predict(kmeanFormat, famille_list, client_index):
    client = [[]]
    for f in famille_list:
        client[0].append(kmeanFormat[f][client_index])
    return client


def fit_kmean_model(kmeanFormat, famille_list):
    df = pd.DataFrame(kmeanFormat, columns = famille_list)
    kmeans = KMeans(n_clusters = 8).fit(df)
    return kmeans


def get_cli_id_list_of_cli_cluster_group(fitedKmeanModel, userPercentByFamilly, client):
    prediction = fitedKmeanModel.predict(client)
    client_to_include = []
    for i in range(len(fitedKmeanModel.labels_)):
        if prediction[0] == fitedKmeanModel.labels_[i]:
            client_to_include.append(userPercentByFamilly[i]['CLI_ID'])
    return client_to_include


############
#
# PUBLIC FUNCTION TO USE
#
############
def get_recommendation(my_client_id, rowCsv=100000):
    metadata = init_dataframe(rowCsv)
    famille_list = metadata["FAMILLE"].unique()

    # getting prefered familly for my_client_id 
    clientMostPopularFamilleDf = most_popular_famille(metadata.loc[metadata['CLI_ID'].isin([my_client_id])])
    print(clientMostPopularFamilleDf)
    clientMostPopularFamille = clientMostPopularFamilleDf["FAMILLE"].tolist()[0]

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
    userPercentByFamilly = get_user_proportion_by_family(metadata, famille_list)

    # getting a model that kmean will work with
    # column : all different familly
    # each index represent the percent of buyed familly of the user[index]
    kmeanFormat = user_proportion_by_famille_to_kmean_format(userPercentByFamilly, famille_list)
    
    # training the model with columns being all different familly
    fitedKmeanModel = fit_kmean_model(kmeanFormat, famille_list)

    # getting the equivalent index in the kmeanFormat of my client_id
    client_index = get_cli_index_from_cli_id(userPercentByFamilly, my_client_id)

    # getting my client in the needed format for the prediction
    myClient = create_client_for_kmean_predict(kmeanFormat, famille_list, client_index)

    # predict the client cluster group then 
    # retrive the list of client_id from the cluster my my_client_id belongs to
    cliIdList = get_cli_id_list_of_cli_cluster_group(fitedKmeanModel, userPercentByFamilly, myClient)
    
    # getting a dataframe with only ticket with client_id in the list
    clientClusterMetadata = metadata.loc[metadata['CLI_ID'].isin(cliIdList)].sort_values(by=['CLI_ID']).reset_index()

    # getting recommandation item based on best score and prefered familly of my client
    recommandation = get_recommendation_strategie_1(clientClusterMetadata, my_client_id, clientMostPopularFamille)
    return recommandation
