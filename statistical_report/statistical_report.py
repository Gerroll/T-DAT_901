import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PDF_PATH = './res/statistical_report.pdf'
PRINT_PDF = False

def numbersOfItems(data):
    """Numbers of items per Maille, Univers and Family """
    univers = data.value_counts(['UNIVERS'])
    maille = data.value_counts(['MAILLE'])
    famille = data.value_counts(['FAMILLE'])
    items = data.value_counts(['LIBELLE'])
    ticket_id = data.value_counts(['TICKET_ID'])
    print("Contenu du jeu de donnée ")
    print("Maille : ", maille.count())
    print("Univers :", univers.count())
    print("Famille : ", famille.count())
    print("Objets différents : ", items.count())
    print("Nombre de paniers: ", ticket_id.count())
    print()



def mostPopularInUnivers(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    popular_items = data.groupby(['UNIVERS', 'LIBELLE']).size().to_frame(name='size').reset_index().sort_values(
        by=['size'], ascending=False)
    print("Objets les plus populaires par univers")
    print(popular_items.drop_duplicates(subset=['UNIVERS']).head())
    print()



def mostPopularInFamille(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    popular_items = data.groupby(['FAMILLE', 'LIBELLE']).size().to_frame(name='size').reset_index().sort_values(
        by=['size'], ascending=False)
    print("Objets les plus populaires par Famille")
    print(popular_items.drop_duplicates(subset=['FAMILLE']).head())
    print()



def meanPriceInFamille(data):
    """ Mean price for items  by Famille """
    mean_price = data.groupby(['FAMILLE'])
    print(mean_price['FAMILLE', "PRIX_NET"].describe())

def meanPriceOfATicket(data):
    """Means price spend on tickets given by the dataset """
    ticket_union = data.groupby(['TICKET_ID'])
    print(ticket_union['PRIX_NET'].mean())


def meanPriceInUnivers(data):
    """ Mean price for items  by Univers """
    mean_price = data.groupby(['UNIVERS'])
    print("Prix des elements pour chaque univers")
    print(mean_price['UNIVERS', "PRIX_NET"].mean().describe())
    print()

    return mean_price

def meanAndStdNumbersOfItemByClients(data):
    """ Mean and std numbers of items per Ticket """
    client_description = data.value_counts(['CLI_ID'])

    print('Nombre de clients : ', client_description.count())
    print("Quantité d'objets par client:", client_description.min(), '-', client_description.max())
    print("Nombre moyen d'objet par client :", client_description.mean())
    print('Std number of items by client : ', client_description.std())
    print()


def meanAndNumbersOfItemsByTicket(data):
    """ Mean and std numbers of items per Ticket """
    item_description = data.value_counts(['TICKET_ID'])
    print('Nombre de paniers : ', item_description.count())
    print("Quantité d'objets par paniers:", item_description.min(), '-', item_description.max())
    print("Nombre moyen d'objet par paniers :", item_description.mean())
    print()

def histPriceByTicket(data):
    """Histogram of x: Price of ticket / y: Number of Ticket"""
    fig = plt.figure()
    priceByTicket = data.groupby("TICKET_ID")["PRIX_NET"].sum()
    #when the range is to high, there is nothing to see
    if priceByTicket.max() <= 250:
        max = priceByTicket.max()
    else:
        max = 250
    # To have a consistant number of bins no matter the quantity of value
    numberOfBins = int(math.log(priceByTicket.size,2))
    plotprice = priceByTicket.hist(bins=numberOfBins,range=[0,max])
    plotprice.set_xlabel("Prix du panier")
    plotprice.set_ylabel("Nombre de paniers")
    plt.suptitle("Nombre de paniers par prix")
    
    if PRINT_PDF is False:
      plt.show()
    plt.close()

    return fig

def histTicketByFamille(data):
    """histogram of every Famille with the y: number of item bought /x : price of the item o"""
    plots = data.hist(by='FAMILLE', column="PRIX_NET")
    for subplots in plots:
        for plot in subplots:
            print(plot)
            plot.set_xlabel("Prix", fontsize=8)
            plot.set_ylabel("Quantités", fontsize=8)
    plt.suptitle('Nombre de produits achetés par famille')

    if PRINT_PDF is False:
      plt.show()
    plt.close()

def histNumberOfTicketByPrice(data):
    """histogram of x: price of ticket/ y : number of ticket"""
    fig = plt.figure()
    month_union = data.groupby(['MOIS_VENTE'])
    month_union['MOIS_VENTE'].hist(bins="auto")
    plt.xticks([1., 3., 5., 7., 9., 11.], ["Janvier", "Mars", "Mai", "Juillet", "Septembre", "Novembre"])
    plt.suptitle('Nombre de produits achetés par Mois')

    if PRINT_PDF is False:
      plt.show()
    plt.close()
    
    return fig

def bestCliForTest(data):
    """return a subset of the data with the cli_id that has the most items buyed in the subset  """
    return data[data['CLI_ID'] == data['CLI_ID'].value_counts().idxmax()]

def getCliData(data, client_id):
    """return a subset of the data with the cli_id specified  """
    if client_id:
      return data[data['CLI_ID'] == client_id]
    else:
      return data

def printData(data):
    """Display values and plot about the dataset"""
    pdf = generatePdf()

    # Histograms
    # fig1 = histTicketByFamille(data)
    fig2 = histNumberOfTicketByPrice(data)
    fig3 = histPriceByTicket(data)

    # save figures for now - TODO save figs to generated pdf
    if PRINT_PDF is True:
      saveFig(fig2, 'histNumberOfTicketByPrice')
      saveFig(fig3, 'histPriceByTicket')
    else:
      # remove the pdf
      removePdf()

    numbersOfItems(data)
    meanAndStdNumbersOfItemByClients(data)
    meanAndNumbersOfItemsByTicket(data)

    mostPopularInUnivers(data)
    mostPopularInFamille(data)

    
def removePdf():
  """Delete the pdf"""
  os.remove(PDF_PATH)

def generatePdf():
  """Create a pdf and return it"""
  return PdfPages(PDF_PATH)

def saveFig(fig, name):
  """Add a fig to the pdf"""
  fig.savefig(f'./res/{name}.png')
