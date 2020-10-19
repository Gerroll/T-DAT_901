import pandas as pd

metadata = pd.read_csv('KaDo.csv', low_memory=False, nrows=500000)

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

def getLibelleScoreDF():
    libelleScore = getScoreForLibelleDF().sort_values(by=['LIBELLE'])
    universLibelle = metadata[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']].copy().groupby(['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']).size().reset_index().sort_values(by=['LIBELLE'])
    return pd.merge(universLibelle[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE']], libelleScore[['LIBELLE', 'SCORE']], on=['LIBELLE'], how='outer').sort_values(by=['SCORE'], ascending=False).reset_index()[['FAMILLE', 'MAILLE', 'UNIVERS', 'LIBELLE', 'SCORE']]

print(getLibelleScoreDF())

# print(metadata)
# print(scoreTable)


def printMeanPriceForLibelle():
    print(metadata[["LIBELLE", "PRIX_NET"]].groupby(["LIBELLE"])["PRIX_NET"].mean().sort_values())

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
