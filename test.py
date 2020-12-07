from enum import Enum

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from timeit import Timer
import time

debut = time.time()
data = pd.read_csv("./res/KaDoSample.csv")
print('after read ', time.time() - debut)
class Column(Enum):
    TICKET_ID = "TICKET_ID"
    MOIS_VENTE = "MOIS_VENTE"
    PRIX_NET = "PRIX_NET"
    FAMILLE = "FAMILLE"
    UNIVERS = "UNIVERS"
    MAILLE = "MAILLE"
    LIBELLE = "LIBELLE"
    CLI_ID = "CLI_ID"


def numbersOfItems(data):
    """Numbers of items per Maille, Univers and Family """
    univers = data.value_counts(['UNIVERS'])
    maille = data.value_counts(['MAILLE'])
    famille = data.value_counts(['FAMILLE'])
    items = data.value_counts(['LIBELLE'])
    ticket_id = data.value_counts(['TICKET_ID'])
    print("Numbers of items ")
    print("Maille : ", maille.count())
    print("Univers :", univers.count())
    print("Famille : ", famille.count())
    print("ITEMS : ", items.count())
    print("TICKED_ID : ", ticket_id.count())


def mostPopularInUnivers(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    familleunivers = data.groupby(['UNIVERS', 'LIBELLE']).size().to_frame(name='size').reset_index().sort_values(
        by=['size'], ascending=False)
    print(familleunivers.drop_duplicates(subset=['UNIVERS']))


def mostPopularInFamille(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    FAMILLEUNIVERS = data.groupby(['FAMILLE', 'LIBELLE']).size().to_frame(name='size').reset_index().sort_values(
        by=['size'], ascending=False)
    print(FAMILLEUNIVERS.drop_duplicates(subset=['FAMILLE']))


def meanPriceInFamille():
    """ Mean price for items  by Famille """
    mean_price = data.groupby(['FAMILLE'])
    print(mean_price['FAMILLE', "PRIX_NET"].describe())


def meanPriceInUnivers():
    """ Mean price for items  by Univers """
    mean_price = data.groupby(['UNIVERS'])
    print(mean_price['UNIVERS', "PRIX_NET"].describe())


def meanAndStdNumbersOfItemByClients(data):
    """ Mean and std numbers of items per Ticket """
    client_description = data.value_counts(['CLI_ID'])

    print('Number of clients : ', client_description.count())
    print('Range of items by client: ', client_description.min(), '-', client_description.max())
    print('Mean number of items by client : ', client_description.mean())
    print('Std number of items by client : ', client_description.std())

    # Full description of the data
    # print(clientDescription.mean())


def meanAndNumbersOfItemsByTicket(data):
    """ Mean and std numbers of items per Ticket """
    item_description = data.value_counts(['TICKET_ID'])

    print('Number of Ticket : ', item_description.count())
    print('Range of items by Ticket: ', item_description.min(), '-', item_description.max())
    print('Mean number of items by Ticket : ', item_description.mean())
    print('Std number of items by Ticket : ', item_description.std())


def meanPriceOfATicket(data):
    """Means price spend on tickets given by the dataset """
    ticket_union = data.groupby(['TICKET_ID'])
    print(ticket_union['PRIX_NET'].mean())


def bestCliForTest():
    """return a subset of the data with the cli_id that has the most items buyed in the subset  """
    # print(data['CLI_ID'].value_counts().idxmax())
    return data[data['CLI_ID'] == data['CLI_ID'].value_counts().idxmax()]


def printData(data):
    numberOfTicketByFamille = data.hist(by='FAMILLE', column="PRIX_NET")
    plt.suptitle('Nombre de produit acheté par famille')
    plt.show()
    month_union = data.groupby(['MOIS_VENTE'])
    print(month_union.describe())
    numberOfTicketByMonth = month_union['MOIS_VENTE'].hist()
    plt.suptitle('Nombre de produit acheté par Mois')
    plt.show()



# numbersOfItems(data=data)
# mostPopularInUnivers(data=bestCliForTest())

debut = time.time()
data = bestCliForTest()
print('after best cli ', time.time() - debut)

debut = time.time()
printData(data=data)
print('after mean Items by ticker ', time.time() - debut)
