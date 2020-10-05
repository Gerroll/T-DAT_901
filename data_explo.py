import pandas as pd

metadata = pd.read_csv('KaDo.csv', low_memory=False, nrows=500000)


def printScoreDF():
    cliId_libelle =  metadata[["CLI_ID", "LIBELLE"]].copy().groupby(["CLI_ID", "LIBELLE"]).size().to_frame(name = 'size').reset_index().sort_values(by=['size'])
    # decrement size colum
    cliId_libelle['size'] -= 1
    # replace nb of buied an item by client to 1
    cliId_libelle["size"].mask(cliId_libelle["size"] >= 1, 1)
    # count all client 
    print(cliId_libelle.groupby(["LIBELLE"]).size().to_frame(name = 'SCORE').reset_index().sort_values(by=['SCORE']))

printScoreDF()

def printMeanPriceForLibelle():
    print(metadata[["LIBELLE", "PRIX_NET"]].groupby(["LIBELLE"])["PRIX_NET"].mean().sort_values())

def simple_count():
    print(metadata.head(1))
    print()
    print(metadata[["PRIX_NET", "LIBELLE"]].copy().groupby(["LIBELLE", "PRIX_NET"]).size())
    print()
    print(len(metadata["LIBELLE"].value_counts(dropna=False)))
    print()
    print(len(metadata["CLI_ID"].value_counts(dropna=False)))
    print()
    print(metadata["LIBELLE"].value_counts(dropna=False))
    print()
    print(metadata["LIBELLE"].value_counts(dropna=False)["FAP PDRE MAUVE CHRYSTAL  2G LUMIN 4  VPM"])
    print()
    print(metadata["CLI_ID"].value_counts(dropna=False))
    print()
    print(len(metadata["UNIVERS"].value_counts(dropna=False)))
    print()
    print(len(metadata["MAILLE"].value_counts(dropna=False)))
    print()
    print(len(metadata["FAMILLE"].value_counts(dropna=False)))
    print()
    print(metadata["MOIS_VENTE"].value_counts(dropna=False))


# occby(user and label) = size
#   |
#   |
#   |
#   |         label ==> same prix ?
#   |           |
#   |           |
#   |           |
#   v           v
# (size - 1 + prix) / prix
#   |
#   |
#   v
# score | label
