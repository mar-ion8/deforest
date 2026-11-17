#import des librairies
import os
import rasterio
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from rasterio.warp import reproject, Resampling
import re

#import des données
data = os.path.join('/assets/data')
lulc = os.path.join(data, 'Mapbiomas_Land_Cover.shp')
landsat = os.path.join(data, 'landsat/')

bandes = {}
for f in Path(landsat).glob("*.TIF"):
    m = re.search(r'(SR_B\d+)', f.stem, re.IGNORECASE)
    if m is None:
        continue
    with rasterio.open(f) as src:
        bandes[m.group(1).lower()] = src.read(1).astype("float32")
        profile = src.profile.copy()
sr_b2 = bandes['sr_b2']
sr_b3 = bandes['sr_b3']
sr_b4 = bandes['sr_b4']
sr_b5 = bandes['sr_b5']
sr_b6 = bandes['sr_b6']
sr_b7 = bandes['sr_b7']

def build_sample() :

    #vérifier la reproj sur Landsat

    #vérifier les noms de champ et décider des champs utilisés

    #sélectionner n polygones par classe
    n = 30

    #récupèrer le centroïd

    #donner au centroïd différentes mesures



build_sample()

def build_stack() :

    #création des indices spectraux
    ndvi = (sr_b5 - sr_b4 ) / (sr_b5 + sr_b4 )
    evi = 2.5 * ((sr_b5 - sr_b4) / (sr_b5 + 6 * sr_b4 - 7.5 * sr_b2 + 1 ))
    ndwi = (sr_b3 - sr_b5) / (sr_b3 + sr_b5)

build_stack()