import math
import matplotlib.pyplot as plt
import pandas as pd
from .pdf import PDF


PDF_PATH = './res/statistical_report.pdf'
PRINT_PDF = True

LABEL_MOIS = ["Janv.", "Fevr.", "Mars", "Avril", "Mai", "Juin", "Juil.", "Aout", "Sept.", "Octob.",
                  "Novem.", "Decem."]
EVENTS = ["Soldes d'hiver", "St-Valentin", "Soldes d'été", "Black Friday", "Cyber Monday", "Noël"]
MOIS_TICKS = [0.,1.,2.,3., 4.,5.,6.,7., 8., 9., 10., 11.]
MOIS_TICKS_HIST = [1.,2.,3., 4.,5.,6.,7., 8., 9., 10., 11.,12.]

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
    popular_items.drop_duplicates(subset=['UNIVERS'])
    print(popular_items)


def mostPopularInFamille(data):
    """For a given Dataset, the most popular items with their libelle and their count  """
    popular_items = data.groupby(['FAMILLE', 'LIBELLE']).size().to_frame(name='size').reset_index().sort_values(
        by=['size'], ascending=False)
    print("Objets les plus populaires par Famille")
    print(popular_items.drop_duplicates(subset=['FAMILLE']).head())
    print()
    plt.show()
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
    string_to_return = ""
    string_to_return += 'Nombre de paniers : '+ item_description.count().__str__()+ '\n';
    string_to_return += "Quantité d'objets par paniers:"+item_description.min().__str__()\
                        + '-' + item_description.max().__str__()+'\n'
    string_to_return +="Nombre moyen d'objet par paniers :" + round(item_description.mean(), 2).__str__()+'\n'
    string_to_return += "\n\n"
    print('Nombre de paniers : ', item_description.count())
    print("Quantité d'objets par paniers:", item_description.min(), '-', item_description.max())
    print("Nombre moyen d'objet par paniers :", item_description.mean())
    return string_to_return
def histPriceByTicket(data,axarr):
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
    plotprice = priceByTicket.hist(bins=numberOfBins, range=[0, max], ax=axarr)
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
    print(sums['PRIX_NET'])
    plt.pie(sums['PRIX_NET'], labels=sums.index, autopct='%1.1f%%',startangle=90)
    plt.axis = 'equal'
    plt.suptitle('Somme dépensé par famille')
    if PRINT_PDF is False:
        plt.show()
        plt.close()

    return fig

def printFamilleMaxSpend(data):
    """Print the famille with the max euros spent """
    print("Max count")
    print(data.value_counts(['FAMILLE']))
    sums = data.groupby('FAMILLE').sum()
    line_to_print = '\n\nFamille préférée  : "' + sums["PRIX_NET"].idxmax()+\
                    '" \nSomme dépensées :  ' + round(sums["PRIX_NET"].max(), 2).__str__()
    print(line_to_print)
    return line_to_print

def printFamilleMaxBought(data):
    """Print the famille with the max euros spent """
    sums = data.cou('FAMILLE').count()
    line_to_print = 'La famille préférée de ce client est "' + sums["PRIX_NET"].idxmax()+\
                    '" avec ' + round(sums["PRIX_NET"].max(), 2).__str__() + "€ dépensés."
    print(line_to_print)

def histNumberOfTicketByMonth(data,axarr):
    """histogram of x: month/ y : number of ticket"""
    month_union = data.groupby(['MOIS_VENTE'])
    month_union['MOIS_VENTE'].hist(bins="auto",ax=axarr)

    axarr.set_xticks(MOIS_TICKS_HIST)
    axarr.set_xticklabels(LABEL_MOIS, fontsize=9)
    axarr.title.set_text('Nombre de produits achetés par Mois')

    if PRINT_PDF is False:
        plt.show()
        plt.close()



def histPricePayedByMonth(data,axarr):
    """histogram of x: sum of price of ticket/ y : number of ticket"""
    fig = plt.figure()
    sums = data.groupby(['MOIS_VENTE'])
    sums['PRIX_NET'].sum().plot(kind='bar',ax=axarr)

    #plt.xticks(MOIS_TICKS, LABEL_MOIS)
    axarr.set_xticks(MOIS_TICKS)
    axarr.set_xticklabels(LABEL_MOIS, fontsize=10)
    axarr.title.set_text('Somme dépensé par Mois')

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
    sums_full = data_full.groupby(['MOIS_VENTE'])['PRIX_NET'].quantile(.80)
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

def compareHistPricePayedByFamille(data_user, data_full):
    """Compare Price spend by Famille between a big dataset (full, cluster) with the data of a user"""
    fig = plt.figure()
    sums = data_user.groupby(['FAMILLE'])['PRIX_NET'].sum()
    sums_full = data_full.groupby(['FAMILLE'])['PRIX_NET'].quantile(.80)
    frame = pd.DataFrame({
        "Utilisateur": sums,
        "Ensemble Du dataset": sums_full
    })
    frame.plot(kind='bar')

    plt.legend(loc='upper right')
    plt.suptitle('Comparaison des achats par famille entre notre utilisateur et le dataset')

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
    # Pie
    figs.append(pieTicketByFamille(data))
    figs.append(piePriceByFamille(data))
    subplot , axarr = plt.subplots(2,1)

    # Histograms
    histNumberOfTicketByMonth(data,axarr[1])
    histPricePayedByMonth(data,axarr[0])

    mostPopularInUnivers(data)
    mostPopularInFamille(data)

    eventText = getEventRelatedToPricePayedByMonth(data)
    eventText2 = meanAndNumbersOfItemsByTicket(data) + printFamilleMaxSpend(data)

    # PDF
    if PRINT_PDF is True:
      pdf = PDF(clientId)

      # save event
      pdf.saveFigInPdf(pdf.textToFig(eventText))
      pdf.saveFigInPdf(pdf.textToFig(eventText2))

      # save our generated figs hists/pies/charts
      print('Save fig into pdf')
      pdf.saveFigInPdf(subplot)
      for fig in figs:
        pdf.saveFigInPdf(fig)


      pdf.closePdf()


def compareResult(data_user, data_full):
    """Compare a big dataset (full, cluster) with the data of a user"""
    compareHistPricePayedByMonth(data_user, data_full)
    compareHistPricePayedByFamille(data_user, data_full)
