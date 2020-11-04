import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv("./res/KaDoSample.csv")
# data['TICKET_ID'] = data['TICKET_ID'].astype('string')
# data['CLI_ID'] = data['CLI_ID'].astype('string')
# data['MOIS_VENTE'] = data['MOIS_VENTE'].astype('object')

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