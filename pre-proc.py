#import des librairies
import os
from pathlib import Path
import re
import rasterio #traitement raster
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn import datasets #modèles de classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np #calculs
import geopandas as gpd #traitements GDF
from tabulate import tabulate #affichage des GDF

#import des données
data = Path(__file__).parent / 'assets' / 'data'
lulc = gpd.read_file(data / 'Mapbiomas_Land_Cover.shp')
landsat = data / 'landsat'

#récupération des bandes Landsat
dst_crs = "EPSG:4326"
bandes = {}
profile = None
for f in Path(landsat).glob("*.TIF"):
    m = re.search(r'(SR_B\d+)', f.stem, re.IGNORECASE)
    if m is None:
        continue
    with rasterio.open(f) as src:
        arr = src.read(1).astype("float32")
        src_crs, src_transform = src.crs, src.transform
        src_bounds, (h, w) = src.bounds, src.shape

    #encodage des valeurs nan + mise à l'échelle du fichier Landsat
    arr[arr == 0] = np.nan
    arr = arr * 0.0000275 - 0.2
    arr[(arr < 0) | (arr > 1)] = np.nan

    #reprojection en 4326
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, *src_bounds)
    arr_4326 = np.full((dst_h, dst_w), np.nan, dtype="float32")
    reproject(source=arr, destination=arr_4326,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    #mise à jour du profile raster
    bandes[m.group(1).lower()] = arr_4326
    if profile is None:
        profile = dict(driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                       crs=dst_crs, transform=dst_transform,
                       width=dst_w, height=dst_h)

sr_b2, sr_b3, sr_b4 = bandes['sr_b2'], bandes['sr_b3'], bandes['sr_b4']
sr_b5, sr_b6, sr_b7 = bandes['sr_b5'], bandes['sr_b6'], bandes['sr_b7']

def build_sample() :
    """
    Pré-traitement du fichier servant à l'échantillonnage

    :return: entité vectorielle ponctuelle qui servira d'entrée au modèle (test/training sample)
    """

    #passage du jeu de données SHP en GDF
    lulc_2016 = gpd.GeoDataFrame(lulc)

    #vérification des noms de champ et décider des champs utilisés
    print(tabulate(lulc_2016[0:10], headers='keys'))
    lulc_2016['id_u'] = range(1, len(lulc_2016) + 1) #génération d'un id unique
    lulc_2016 = lulc_2016.drop(['count', 'layer', 'path'], axis = 1) #suppression des colonnes non pertinentes
    print(tabulate(lulc_2016[0:10], headers='keys'))

    """L'affichage des champs et leur nettoyage permet d'identifier les champs qui vont être utilisés lors de l'analyse.
    Pour la classification et la référence, le champ "class" sera utilisé pour identifier la classe."""

    print(lulc_2016['class'].unique()) #permet d'identifier le nombre et l'intitulé des classes

    #sélection de n polygones par classe
    lulc_rand = (
        lulc_2016
        .groupby('class', group_keys=False)
        .apply(lambda x: x.sample(n=min(20, len(x)), random_state=42))
    )
    print('nombre de lignes (attendues : inférieur à 300) = ', len(lulc_rand ))
    print(tabulate(lulc_rand [0:10], headers='keys'))

    #récupèration du centroïd
    lulc_rand['centroid'] = lulc_rand.geometry.centroid
    lulc_rand[['id_u', 'class', 'centroid']].to_file(data / "centroids.shp")

    return lulc_rand

build_sample()

def build_stack() :

    """
    Création d'un stack contenant toutes les entités raster servant à la classification

    :return: stack contenant les bandes et indices spectraux pour l'entrée du modèle
    """

    #calcul des indices spectraux
    ndvi = (sr_b5 - sr_b4 ) / (sr_b5 + sr_b4 )
    evi = 2.5 * ((sr_b5 - sr_b4) / (sr_b5 + 6 * sr_b4 - 7.5 * sr_b2 + 1 ))
    ndwi = (sr_b3 - sr_b5) / (sr_b3 + sr_b5)

    #création d'un stack contenant toutes les bandes + les indices spectraux
    stack = [sr_b2, sr_b3, sr_b4, sr_b5, sr_b6, sr_b7, ndvi, evi, ndwi]
    stack = np.stack(stack)

    return stack

build_stack()

def model_rf() :

    """
    Paramétrage et lancement de la classification : modèle de classification supervisée Random Forest

    :return:
    """

    #préparation de l'échantillon
    n=20

#model_rf()