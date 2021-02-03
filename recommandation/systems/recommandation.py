import pandas as pd
from sklearn.cluster import KMeans
from enum import Enum
from pathlib import Path

"""
    Paths
"""
# path to project directory
project_dir = Path(__file__).parent.parent.parent
# path to processed data
processed_dir = project_dir.joinpath('assets').joinpath('processed')
recommendation_proc_file = processed_dir.joinpath('recommendation_proc.pkl')
recommendation_clust_file = processed_dir.joinpath('recommendation_clust.pkl')
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


#
# RECOMANDATION BY SCORE AND FAMILLY PREFERENCE
#
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


def get_cli_id_list_of_cli_cluster_group(predictedDf, cli_id):
    cluster_label = predictedDf[predictedDf['CLI_ID'] == cli_id]['cluster'].iloc[0]
    return list(predictedDf[predictedDf['cluster'] == cluster_label]['CLI_ID'])


#
# PUBLIC FUNCTION TO USE
#
def get_recommendation(my_client_id):
    metadata = pd.read_csv(kado_file)
    famille_list = metadata["FAMILLE"].unique()

    if recommendation_clust_file.is_file():
        # load from previous computing
        predictedDf = pd.read_pickle(recommendation_clust_file)
    else:
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

        # build dataframe from model
        kmeanFormat = user_proportion_by_famille_to_kmean_format(userPercentByFamilly, famille_list)
        predictedDf = pd.DataFrame(kmeanFormat)

        # keep only famille column (without CLI_ID)
        familyProportionDf = pd.DataFrame()
        for f in famille_list:
            familyProportionDf[f] = predictedDf[f]

        # fit kmeans
        kmeans = KMeans(n_clusters=8, random_state=1).fit(familyProportionDf)

        # store prediction into dataframe
        predictedDf['cluster'] = kmeans.labels_

        # save prediction to pickle
        predictedDf.to_pickle(recommendation_clust_file)

    # retrieve the list client id from the cluster of the current user id
    cliIdList = get_cli_id_list_of_cli_cluster_group(predictedDf, my_client_id)
    
    # getting a dataframe with only ticket with client_id in the list
    clientClusterMetadata = metadata.loc[metadata['CLI_ID'].isin(cliIdList)].sort_values(by=['CLI_ID']).reset_index()

    # getting prefered familly for my_client_id
    clientMostPopularFamilleDf = most_popular_famille(metadata.loc[metadata['CLI_ID'].isin([my_client_id])])
    clientMostPopularFamille = clientMostPopularFamilleDf["FAMILLE"].tolist()[0]

    # getting recommandation item based on best score and prefered familly of my client
    recommandation = get_recommendation_strategie_1(clientClusterMetadata, my_client_id, clientMostPopularFamille)
    return recommandation


if __name__ == "__main__":
    print(get_recommendation(1490281))
