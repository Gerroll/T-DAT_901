# T-DAT_901
### Recommender, outils pour recomander a partir de donnée utilisateur

# Init

`Récupérer "KaDo.csv" sur gandalf, à mettre a la racine du project`


# Ex: csv file of 7 million lines (KaDo.csv)

## Formater pour .md :

|TICKET_ID|MOIS_VENTE|PRIX_NET|FAMILLE|UNIVERS|MAILLE|LIBELLE|CLI_ID|
|---------|----------|--------|-------|-------|------|-------|-------|
|35592159|10|1.67|HYGIENE|HYG_DOUCHE JARDINMONDE|HYG_JDM|GD JDM4 PAMPLEMOUSSE FL 200ML|1490281|
|35592159|10|1.66|HYGIENE|HYG_DOUCHE JARDINMONDE|HYG_JDM|GD JDM4 PAMPLEMOUSSE FL 200ML|1490281|
|35592159|10|7.45|SOINS DU VISAGE|VIS_CJOUR Jeunes Specifique|VIS_JEUNE_ET_LEVRE|CR JR PARF BIO.SPE AC.SENT.50ML|1490281|

<br/>

## Non formater :

TICKET_ID,MOIS_VENTE,PRIX_NET,FAMILLE,UNIVERS,MAILLE,LIBELLE,CLI_ID
35592159,10,1.67,HYGIENE,HYG_DOUCHE JARDINMONDE,HYG_JDM,GD JDM4 PAMPLEMOUSSE FL 200ML,1490281
35592159,10,1.66,HYGIENE,HYG_DOUCHE JARDINMONDE,HYG_JDM,GD JDM4 PAMPLEMOUSSE FL 200ML,1490281
35592159,10,7.45,SOINS DU VISAGE,VIS_CJOUR Jeunes Specifique,VIS_JEUNE_ET_LEVRE,CR JR PARF BIO.SPE AC.SENT.50ML,1490281


## Using Virtualenv
### Install virtual env
`pip install virtualenv`

### Init virtualenv
`virtualenv -p python3.7 venv3.7`

### Use existing virtualenv
Ubuntu : 
`source venv3.7/bin/activate`

Windows: 
`venv3.7\Scripts\activate.bat`

### Install package in virtualenv
`pip install <module>`

### Save dependencies
`pip freeze > requirements3.7.txt`

### Load dependencies
`pip install -r requirements3.7.txt`

### Stop using virtualenv
`deactivate`

## Using [Pipenv](https://pipenv.pypa.io/en/latest/)
### Install pipenv
`pip install --user pipenv`

### Init project with python 3.7
`pipenv --python path/to/python3.7`

### Install dependencies from requirements.txt 
`pipenv install -r requirements.txt`

### Install package
`pipenv install <module>`

### Activate environnement
`pipenv shell`

### Run program
`pipenv run python3 app.py`
