# T-DAT_901
### Recommender, outils pour recomander a partir de donnée utilisateur

# Init

`Récupérer "KaDo.csv" sur gandalf, à mettre dans le dossier assets/resource`

##Description de la data
####Colonnes du CSV 
- FAMILLE        object
- TICKET_ID       int64
- MOIS_VENTE      int64
- PRIX_NET      float64
- UNIVERS        object
- MAILLE         object
- LIBELLE        object
- CLI_ID          int64
# Ex: csv file of 7 million lines (KaDo.csv)

## Formatées  pour .md :

|TICKET_ID|MOIS_VENTE|PRIX_NET|FAMILLE|UNIVERS|MAILLE|LIBELLE|CLI_ID|
|---------|----------|--------|-------|-------|------|-------|-------|
|35592159|10|1.67|HYGIENE|HYG_DOUCHE JARDINMONDE|HYG_JDM|GD JDM4 PAMPLEMOUSSE FL 200ML|1490281|
|35592159|10|1.66|HYGIENE|HYG_DOUCHE JARDINMONDE|HYG_JDM|GD JDM4 PAMPLEMOUSSE FL 200ML|1490281|
|35592159|10|7.45|SOINS DU VISAGE|VIS_CJOUR Jeunes Specifique|VIS_JEUNE_ET_LEVRE|CR JR PARF BIO.SPE AC.SENT.50ML|1490281|

<br/>

## Non formatées  :

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
`pip3 install <module>`

### Save dependencies
`pip3 freeze > requirements3.7.txt`

### Load dependencies
`pip3 install -r requirements3.7.txt`
sudo can be needed

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
