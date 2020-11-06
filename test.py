import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from timeit import Timer
import time

debut = time.time()
data = pd.read_csv("./res/KaDo.csv" )
print('after read ' , time.time() - debut)
#data.sort_values(['LIBELLE'])
# data['TICKET_ID'] = data['TICKET_ID'].astype('string')
# data['CLI_ID'] = data['CLI_ID'].astype('string')
# data['MOIS_VENTE'] = data['MOIS_VENTE'].astype('object')
"""
print(data.info())
CLI_ID = data.value_counts(['CLI_ID'])
PRIX_NET = data.value_counts(['PRIX_NET'])
PRIX_NET.sort_index(inplace=True)

test = data.groupby('FAMILLE')['PRIX_NET', 'CLI_ID']
test.plot.scatter(x='PRIX_NET', y='CLI_ID',stacked=True, s=10)
plt.show()




print(test)
LIBELLE = data.value_counts(['LIBELLE'])
MOIS_VENTE = data.value_counts(['MOIS_VENTE'])
UNIVERS = data.value_counts(['UNIVERS'])
MOIS_VENTE.sort_index(inplace=True)
FAMILLE =  data.value_counts(['FAMILLE'])

CLI_ID =data.value_counts(['CLI_ID'])
print(CLI_ID)

CLI_ID.cumsum()

print(CLI_ID)
#print(PRIX_NET.head(5))
#print(UNIVERS)
# CLI_ID.plot()
# PRIX_NET.plot()
# LIBELLE.plot()
# MOIS_VENTE.plot()


plt.subplot(211)
MOIS_VENTE.plot(kind='bar')

plt.subplot(212)
PRIX_NET.plot(subplots=True)
plt.show()

plt.subplot(111)
UNIVERS.head(20).plot(kind='barh')
plt.show()

plt.subplot(111)
FAMILLE.plot(kind='pie')
plt.show()

plt.subplot(111)
CLI_ID.plot()
plt.show()
"""

def numbersOfItems(data):
    """Numbers of items per Maille, Univers and Family """
    UNIVERS = data.value_counts(['UNIVERS'])
    MAILLE = data.value_counts(['MAILLE'])
    FAMILLE = data.value_counts(['FAMILLE'])
    ITEMS  = data.value_counts(['LIBELLE'])
    TICKET_ID  = data.value_counts(['TICKET_ID'])

    print("Maille : ", MAILLE.count())
    print("Univers :", UNIVERS.count())
    print ("Famille : ", FAMILLE.count())
    print ("ITEMS : ", ITEMS.count())
    print("TICKED_ID : ", TICKET_ID.count())

    print("Numbers of items ")

   # FAMILLEUNIVERS = data.groupby(['LIBELLE'])#.value_counts(['LIBELLE'])
    #test = FAMILLEUNIVERS.head()
    #print(FAMILLEUNIVERS.describe())

    #print(test.value_counts(['MAILLE','LIBELLE']))
    #print(test.value_counts(['UNIVERS']))
    #print(test.value_counts(['FAMILLE']))



def mostPopularInUnivers(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    FAMILLEUNIVERS = data.groupby(['UNIVERS','LIBELLE']).size().to_frame(name='size').reset_index().sort_values(by=['size'],ascending=False)
    print(FAMILLEUNIVERS.drop_duplicates(subset=['UNIVERS']))

def mostPopularInFamille(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    FAMILLEUNIVERS = data.groupby(['FAMILLE','LIBELLE']).size().to_frame(name='size').reset_index().sort_values(by=['size'],ascending=False)
    print(FAMILLEUNIVERS.drop_duplicates(subset=['FAMILLE']))


def meanPriceInFamille():
    """ Mean price for items  by Famille """
    meanPrice = data.groupby(['FAMILLE'])
    print(meanPrice['FAMILLE', "PRIX_NET"].describe())

def meanPriceInUnivers():
    """ Mean price for items  by Univers """
    meanPrice = data.groupby(['UNIVERS'])
    print(meanPrice['UNIVERS', "PRIX_NET"].describe())

def meanAndStdNumbersOfItemByClients(data):
    """ Mean and std numbers of items per Ticket """
    clientDescription = data.value_counts(['CLI_ID'])

    print('Number of clients : ', clientDescription.count())
    print('Range of items by client: ', clientDescription.min() , '-',  clientDescription.max())
    print('Mean number of items by client : ',clientDescription.mean())
    print('Std number of items by client : ',clientDescription.std())

    #Full description of the data
    #print(clientDescription.mean())


def meanAndNumbersOfItemsByTicket(data):
    """ Mean and std numbers of items per Ticket """
    itemDescription = data.value_counts(['TICKET_ID'])

    print('Number of Ticket : ',itemDescription.count())
    print('Range of items by Ticket: ',itemDescription.min(),'-',itemDescription.max())
    print('Mean number of items by Ticket : ',itemDescription.mean())
    print('Std number of items by Ticket : ',itemDescription.std())

    #Full description of the data
    # print(itemDescription.describe())

def meanPriceOfATicket(data):
    """Means price spend on tickets given by the dataset """
    ticketUnion = data.groupby(['TICKET_ID'])
   # ticketUnion.dropna(subset=['PRIX_NET'])
    print(ticketUnion['PRIX_NET'].mean())


def bestCliForTest():
    """return a subset of the data with the cli_id that has the most items buyed in the subset  """
    #print(data['CLI_ID'].value_counts().idxmax())
    return data[data['CLI_ID'] == data['CLI_ID'].value_counts().idxmax()]


#numbersOfItems(data=data)
#mostPopularInUnivers(data=bestCliForTest())

debut = time.time()
data = bestCliForTest()
print('after best cli ' , time.time() - debut)


debut = time.time()
mostPopularInFamille(data=data)
print('after mean Items by ticker ' , time.time() - debut)
#meanAndStdNumbersOfItemByClients(data=bestCliForTest())
meanPriceOfATicket(data=data)

