from datetime import time
from textwrap import dedent
import pandas as pd

from extra_description import Clusterer, Clusterer2
N_ROWS = 10000
DEMO_IDS = [1490281, 13290776, 20163348, 20200041, 20561854, 20727324, 20791601, 21046542, 21239163,
            21351166, 21497331, 21504227, 21514622, 69813934, 71891681, 85057203]

def initDataFrame():
    """Get a dataframe with nrows entries """
    if N_ROWS == 0 or N_ROWS == None:
      return pd.read_csv('./data/KaDo.csv', low_memory=False)  # the whole dataset
    else:
      return pd.read_csv('./data/KaDo.csv', low_memory=False, nrows=N_ROWS)# limited rows dataset

#def recommend(metadata, clientId):
#    recomandation = getRecomandation(clientId, metadata)
 #   print(recomandation)


#def printStatisticalReport(metadata, clientId):
    ###
    ### For performance test remove the commentary
    # debut = time.time()

    #whole dataset
    #printData(metadata, clientId)


    # Only with the best client
    #printData(bestCliForTest(metadata))

    # With the specified client id
    #printData(getCliData(metadata, clientId))

    # print('Performance test print Data  : ', time.time() - debut)
"""Initialisation of both clusters"""
def initClusters(raw_data):
    return Clusterer(raw_data), Clusterer2(raw_data)

if __name__ == "__main__":
    # load metadatas from dataframe
    metadata = initDataFrame()

    clusterer, clusterer2 = initClusters(metadata)

    ### Clusters Demo
    for index in DEMO_IDS:
        print('ID : ', index)
        stringClusterer = clusterer.get_description(index)
        stringClusterer2 = clusterer2.get_description(index)
        print("\n\n")
    # ask for the client ID
    print(dedent("""
           This application will give you a recomandation and a statistical report base on your client ID.
           What is your client ID ? (Press 'enter' to get default id)\
       """))
    clientId = input()
    if clientId == "":
        clientId = 1490281
    print(f"Your id : '{clientId}'")
    print("Please Wait ...")

#    recomandation = getRecomandation(clientId, rowCsv=100000)
#    print(recomandation)



    # print stastical report
    #printStatisticalReport(metadata, clientId)
    #compareResult(getCliData(metadata,1490281), metadata)

    # TODO 
    # TODO statisticalReport = getStatisticalReport(clientId)
    # TODO print(statisticalReport)
    # TODO 
