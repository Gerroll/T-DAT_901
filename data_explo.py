import pandas as pd

from enum import Enum

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



metadata = pd.read_csv('KaDo.csv', low_memory=False, nrows=500000)

# just for the sick of it
def printBasicData():
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

# Score definition: Number of client that buy the article at least twice
def getScoreForLibelleDF():
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
def getLibelleScoreDFWithCategories():
    libelleScore = getScoreForLibelleDF().sort_values(by=['LIBELLE'])
    universLibelle = metadata[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']].copy().groupby(['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index().sort_values(by=['LIBELLE'])
    return pd.merge(universLibelle[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']], libelleScore[['LIBELLE', 'SCORE']], on=['LIBELLE'], how='outer').sort_values(by=['SCORE'], ascending=False).reset_index()[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE', 'SCORE']]

def getFirstLineLibelleOfSpecificCategory(libelleScoreDFWithCategories, category: Category, nameCategory: str):
    return libelleScoreDFWithCategories.loc[libelleScoreDFWithCategories[category.name] == nameCategory][:1]

def getListOfLibelleOfTheClientDidNotBuy():
    pass

# return a list of string that represent buying item of one client
def getListOfLibelleOfTheClientDidBuy(client_id):
    df = metadata.copy()
    df.loc[df[Column.CLI_ID.name] == client_id]
    pass

print(metadata[:3])

getListOfLibelleOfTheClientDidBuy("1490281")

# print(getLibelleScoreDFWithCategories()[:5])

# libelleScoreDFWithCategories = getLibelleScoreDFWithCategories()
# print(getFirstLineLibelleOfSpecificCategory(libelleScoreDFWithCategories, Category.FAMILLE, "MAQUILLAGE"))
# print(getFirstLineLibelleOfSpecificCategory(libelleScoreDFWithCategories, Category.MAILLE, "CORPS_HYDRA_NOURRI_ET_SOINS"))
# print(getFirstLineLibelleOfSpecificCategory(libelleScoreDFWithCategories, Category.UNIVERS, "VIS_DEMAQ BLEUET"))
