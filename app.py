from datetime import time
from textwrap import dedent

import pandas as pd

from recommandation import getRecomandation
from statistical_report import printData , bestCliForTest



def initDataFrame(nrows=100000):
    """Get a dataframe with nrows entries """
    metadata = pd.read_csv('./res/KaDoSample.csv', low_memory=False, nrows=nrows)
    return metadata

def recommend(metadata):
    print(dedent("""
           This application will give you a recomandation and a statistical report base on your client ID.
           What is your client ID ? (Press 'enter' to get default id)\
       """))
    clientId = input()
    if clientId == "":
        clientId = 1490281
    print(f"Your id : '{clientId}'")
    print("Please Wait ...")
    recomandation = getRecomandation(clientId, metadata)
    print(recomandation)


def printStatisticalReport(metadata):
    ###
    ### For performance test remove the commentary
    # debut = time.time()

    #whole dataset
    printData(metadata)

    # Only with the best client
    printData(bestCliForTest(metadata))

    # print('Performance test print Data  : ', time.time() - debut)


if __name__ == "__main__":
    metadata = initDataFrame(nrows=10000)
    #recommend(metadata)
    printStatisticalReport(metadata)




