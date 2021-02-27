from textwrap import dedent
import pandas as pd
import sys

from statistical_report import printData
from segmentation import segmentation
from recommandation import rsmerger

N_ROWS = None
DEMO_IDS = [13290776, 20163348, 20200041, 20561854, 20727324, 20791601, 21046542, 21239163,
            21351166, 21497331, 21504227, 21514622, 69813934, 71891681, 85057203]


def run():
    while True:
        print((dedent("""
           What is your client ID ? (Press 'enter' to get default id)\
        """)))
        clientId = input()

        if clientId == "":
            clientId = "13290776"

        try:
            clientId = int(clientId)
        except ValueError:
            print(f"{clientId} is not a valid ID.")
            continue
        if clientId in list_id:
            break

    print(f"Your id : '{clientId}'")
    print("Please Wait ...")

    printData(raw_df, clientId)
    clusterer.display_segmentation(clientId)

    while True:
        print(dedent("""
            Choose your recommendation system:
            1 - clusterbased1
            2 - clusterbased2
            3 - userbased
            4 - ALL
        """))
        rs = input()

        try:
            rs = int(rs)
        except ValueError:
            print(f"{rs} is not a valid choice.")
            continue
        if rs in [1, 2, 3, 4]:
            break

    print("Start computing recommendation...")
    recommendations = {}
    if rs == 1:
        recommendations = merger.get_recommendation(int(clientId), rsmerger.RecommendationType.CLUSTER_BASED_1)
    elif rs == 2:
        recommendations = merger.get_recommendation(int(clientId), rsmerger.RecommendationType.CLUSTER_BASED_2)
    elif rs == 3:
        recommendations = merger.get_recommendation(int(clientId), rsmerger.RecommendationType.USER_BASED)
    elif rs == 4:
        recommendations = merger.get_recommendation(int(clientId), rsmerger.RecommendationType.ALL)

    print("")
    print(f"The item recommended for the client {clientId} is:")
    explanation = recommendations["explanation"]
    print(explanation)


if __name__ == "__main__":
    # load metadatas from dataframe
    try:
        raw_df = pd.read_csv('./data/KaDo.csv')
    except FileNotFoundError:
        print("INFO: You must add KaDo.csv file to ./data folder")
        sys.exit(0)

    list_id = list(raw_df['CLI_ID'].unique())
    clusterer = segmentation.Clusterer()
    merger = rsmerger.Merger()

    # ask for the client ID
    print(dedent("""
           This application will give you a statistical report, a customer's profile and a recommendation for the client's ID given.
       """))
    run()

    while True:
        print(dedent("""
            Make another recommendation ? y/n
        """))

        resp = input()
        if resp == 'y':
            run()
        elif resp == 'n':
            break
