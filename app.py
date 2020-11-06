from textwrap import dedent

from recommandation import getRecomandation

if __name__ == "__main__":
    print(dedent("""
        This application will give you a recomandation and a statistical report base on your client ID.
        What is your client ID ? (Press 'enter' to get default id)\
    """))
    clientId = input()
    if clientId == "":
        clientId = 1490281
    print(f"Your id : '{clientId}'")
    print("Please Wait ...")

    recomandation = getRecomandation(clientId, rowCsv=100000)
    print(recomandation)

    # TODO 
    # TODO statisticalReport = getStatisticalReport(clientId)
    # TODO print(statisticalReport)
    # TODO 
