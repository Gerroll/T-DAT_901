import math
import os
import matplotlib.pyplot as plt
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from .pdf import PDF


PDF_PATH = './res/statistical_report.pdf'
PRINT_PDF = False

LABEL_MOIS = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Aout", "Septembre", "Octobre",
                  "Novembre", "Decembre"]
EVENTS = ["Soldes d'hiver", "St-Valentin", "Soldes d'été", "Black Friday", "Cyber Monday", "Noël"]
MOIS_TICKS = [0.,1.,2.,3., 4.,5.,6.,7., 8., 9., 10., 11.]
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
    # when the range is to high, there is nothing to see
    if priceByTicket.max() <= 250:
        max = priceByTicket.max()
    else:
        max = 250
    # To have a consistant number of bins no matter the quantity of value
    numberOfBins = int(math.log(priceByTicket.size, 2))
    if numberOfBins == 0:
        numberOfBins = 1
    plotprice = priceByTicket.hist(bins=numberOfBins, range=[0, max])
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


def pieTicketByFamille(data):
    """piechart of quantity of sales in every Famille """
    fig = plt.figure()
    sums = data.value_counts('FAMILLE')
    plt.pie(sums, labels=sums.index, autopct='%1.1f%%')
    plt.axis = 'equal'
    plt.suptitle('Nombre de produits achetés par famille')

    if PRINT_PDF is False:
        plt.show()
        plt.close()

    return fig


def piePriceByFamille(data):
    """piechart of volume of price in every Famille for a given dataset """
    fig = plt.figure()
    sums = data.groupby('FAMILLE').sum()
    print(sums)
    plt.pie(sums['PRIX_NET'], labels=sums.index, autopct='%1.1f%%',startangle=90)
    plt.axis = 'equal'
    plt.suptitle('Somme dépensé par famille')
    if PRINT_PDF is False:
        plt.show()
        plt.close()

    return fig

def printFamilleMaxSpend(data):
    """Print the famille with the max euros spent """
    sums = data.groupby('FAMILLE').sum()
    line_to_print = 'La famille préférée de ce client est "' + sums["PRIX_NET"].idxmax()+\
                    '" avec ' + round(sums["PRIX_NET"].max(), 2).__str__() + "€ dépensés."
    print(line_to_print)

def histNumberOfTicketByMonth(data):
    """histogram of x: month/ y : number of ticket"""
    fig = plt.figure()
    month_union = data.groupby(['MOIS_VENTE'])
    month_union['MOIS_VENTE'].hist(bins="auto")
    plt.xticks([1., 4., 7., 10., 12.], ["Janvier", "Avril", "Juillet", "Octobre", "Decembre"])
    plt.suptitle('Nombre de produits achetés par Mois')

    if PRINT_PDF is False:
        plt.show()
        plt.close()

    return fig


def histPricePayedByMonth(data):
    """histogram of x: sum of price of ticket/ y : number of ticket"""
    fig = plt.figure()
    sums = data.groupby(['MOIS_VENTE'])
    sums['PRIX_NET'].sum().plot(kind='bar')

    plt.xticks(MOIS_TICKS, LABEL_MOIS)

    plt.suptitle('Somme dépensé par Mois')

    if PRINT_PDF is False:
        plt.show()
        plt.close()

    return fig

def getEventRelatedToPricePayedByMonth(data):

  sums = data.groupby(['MOIS_VENTE']).sum().sort_values(by=['PRIX_NET'], ascending=False)
  twoBestMonths = sums.take([0, 1])

  firstMonth = int(pd.DataFrame(twoBestMonths)['TICKET_ID'].keys()[0])
  secondMonth = int(pd.DataFrame(twoBestMonths)['TICKET_ID'].keys()[1])

  text = f"L'utilisateur dépense beaucoup aux mois\nde {LABEL_MOIS[firstMonth-1]} et {LABEL_MOIS[secondMonth-1]} pour :\n"
  if isSoldeHiver(firstMonth) or isSoldeHiver(secondMonth): # solde hiver
    text = text + f"- {EVENTS[0]}\n"
  if firstMonth == 2 or secondMonth == 2: # st valentin
    text = text + f"- {EVENTS[1]}\n"
  if isSoldeEte(firstMonth) or isSoldeEte(secondMonth): # solde été
    text = text + f"- {EVENTS[2]}\n"
  if firstMonth == 11 or secondMonth == 11: # black friday - Cyber Monday
    text = text + f"- {EVENTS[3]}\n"
    text = text + f"- {EVENTS[4]}\n"
  if firstMonth == 12 or secondMonth == 12: # noel
    text = text + f"- {EVENTS[5]}\n"

  print(text)
  return text

def isSoldeHiver(month):
    return month == 1 or month == 2
def isSoldeEte(month):
    return month == 6 or month == 7

def compareHistPricePayedByMonth(data_user, data_full):
    """Compare Price spend by month between a big dataset (full, cluster) with the data of a user"""
    fig = plt.figure()
    sums = data_user.groupby(['MOIS_VENTE'])['PRIX_NET'].sum()
    sums_full = data_full.groupby(['MOIS_VENTE'])['PRIX_NET'].mean()
    frame = pd.DataFrame({
        "Utilisateur": sums,
        "Ensemble Du dataset": sums_full
    })
    frame.plot(kind='bar')
    plt.xticks(MOIS_TICKS, LABEL_MOIS)

    plt.legend(loc='upper right')
    plt.suptitle('Comparaison des achats mensuels entre notre utilisateur et le dataset')

    if PRINT_PDF is False:
        plt.show()
        plt.close()

    return fig


def bestCliForTest(data):
    """return a subset of the data with the cli_id that has the most items buyed in the subset  """
    return data[data['CLI_ID'] == data['CLI_ID'].value_counts().idxmax()]


def getCliData(data, clientId):
    """return a subset of the data with the cli_id specified  """
    if clientId:
        return data[data['CLI_ID'] == clientId]
    else:
        return data


def printData(data, clientId):
    """Display values and plot about the dataset"""

    figs = []  # our array of generated figs
    printFamilleMaxSpend(data)
    # Pie
    figs.append(pieTicketByFamille(data))
    figs.append(piePriceByFamille(data))

    # Histograms
    figs.append(histNumberOfTicketByMonth(data))
    figs.append(histPriceByTicket(data))
    figs.append(histPricePayedByMonth(data))

    numbersOfItems(data)
    meanAndStdNumbersOfItemByClients(data)
    meanAndNumbersOfItemsByTicket(data)

    mostPopularInUnivers(data)
    mostPopularInFamille(data)

    eventText = getEventRelatedToPricePayedByMonth(data)

    # PDF
    if PRINT_PDF is True:
      pdf = PDF(clientId)

      # save event
      pdf.saveFigInPdf(pdf.textToFig(eventText))

      # save our generated figs hists/pies/charts
      print('Save fig into pdf')
      for fig in figs:
        pdf.saveFigInPdf(fig)


      pdf.closePdf()


def compareResult(data_user, data_full):
    """Compare a big dataset (full, cluster) with the data of a user"""
    compareHistPricePayedByMonth(data_user, data_full)
