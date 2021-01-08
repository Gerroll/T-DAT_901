from datetime import time
from textwrap import dedent

import pandas as pd

from recommandation import getRecomandation
from statistical_report import printData , bestCliForTest, getCliData,compareResult

N_ROWS = 10000

def initDataFrame():
    """Get a dataframe with nrows entries """
    if N_ROWS == 0 or N_ROWS == None:
      return pd.read_csv('./res/KaDoSample.csv', low_memory=False)  # the whole dataset
    else:
      return pd.read_csv('./res/KaDoSample.csv', low_memory=False, nrows=N_ROWS)# limited rows dataset

def recommend(metadata, clientId):
    recomandation = getRecomandation(clientId, metadata)
    print(recomandation)


def printStatisticalReport(metadata, clientId):
    ###
    ### For performance test remove the commentary
    # debut = time.time()

    #whole dataset
    printData(metadata)


    # Only with the best client
    #printData(bestCliForTest(metadata))

    # With the specified client id
    #printData(getCliData(metadata, clientId))

    # print('Performance test print Data  : ', time.time() - debut)


if __name__ == "__main__":
    # load metadatas from dataframe
    metadata = initDataFrame()
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

    # recommendation system
    #recommend(metadata, clientId)

    # print stastical report
    printStatisticalReport(metadata, clientId)
    #compareResult(bestCliForTest(metadata),metadata)

    # generate pdf




