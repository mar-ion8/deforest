#import des librairies
import os
from pathlib import Path
import re
import rasterio #traitement raster
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np #calculs
import geopandas as gpd #traitements GDF
import pandas as pd
from tabulate import tabulate #affichage des GDF
from shapely.geometry import box
import xarray as xr
import rioxarray  # active l'accessor .rio
from rioxarray.exceptions import NoDataInBounds

#import des données
data = Path(__file__).parent / 'assets' / 'data'
lulc = gpd.read_file(data / 'mapbiomas_lulc_2016.shp')
landsat = data / 'landsat'

NOMS = ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'ndvi', 'evi', 'ndwi']
DST_CRS = "EPSG:4326"

def build_stack() :

    """
    Création d'un stack contenant toutes les entités raster servant à la classification

    :return: stack contenant les bandes et indices spectraux pour l'entrée du modèle
    """
    # récupération des bandes Landsat
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

        # encodage des valeurs nan + mise à l'échelle du fichier Landsat
        arr[arr == 0] = np.nan
        arr = arr * 0.0000275 - 0.2
        arr[(arr < 0) | (arr > 1)] = np.nan

        # reprojection en 4326
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w, h, *src_bounds)
        arr_4326 = np.full((dst_h, dst_w), np.nan, dtype="float32")
        reproject(source=arr, destination=arr_4326,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        transform = dst_transform

        # mise à jour du profile raster
        bandes[m.group(1).lower()] = arr_4326
        if profile is None:
            profile = dict(driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                           crs=dst_crs, transform=dst_transform,
                           width=dst_w, height=dst_h)

    sr_b2, sr_b3, sr_b4 = bandes['sr_b2'], bandes['sr_b3'], bandes['sr_b4']
    sr_b5, sr_b6, sr_b7 = bandes['sr_b5'], bandes['sr_b6'], bandes['sr_b7']

    #calcul des indices spectraux
    ndvi = (sr_b5 - sr_b4 ) / (sr_b5 + sr_b4 )
    evi = 2.5 * ((sr_b5 - sr_b4) / (sr_b5 + 6 * sr_b4 - 7.5 * sr_b2 + 1 ))
    ndwi = (sr_b3 - sr_b5) / (sr_b3 + sr_b5)

    #création d'un stack contenant toutes les bandes + les indices spectraux
    arr = np.stack([sr_b2, sr_b3, sr_b4, sr_b5, sr_b6, sr_b7, ndvi, evi, ndwi])
    arr[~np.isfinite(arr)] = np.nan

    # coordonnées pixel (centres) depuis transform
    _, h, w = arr.shape
    xs = transform.c + transform.a * (np.arange(w) + 0.5)
    ys = transform.f + transform.e * (np.arange(h) + 0.5)

    stack = xr.DataArray(arr, dims=("band", "y", "x"),
                         coords={"band": NOMS, "y": ys, "x": xs})
    stack = stack.rio.write_crs(DST_CRS).rio.write_nodata(np.nan)
    return stack

    return stack

stack = build_stack()

def stats_polygone(geom, stack):
    """Moy, max, vote majoritaire par bande sous un polygone."""
    try:
        xd = stack.rio.clip([geom], crs=DST_CRS, drop=True, all_touched=True)
    except NoDataInBounds:
        return None

    vals = xd.values.reshape(xd.shape[0], -1)   # (n_bandes, n_pixels)
    res = {'px_total': vals.shape[1], 'px_valid': int((~np.isnan(vals[0])).sum())}

    for i, nom in enumerate(NOMS):
        v = vals[i]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            res[f'moy_{nom}'] = res[f'max_{nom}'] = res[f'maj_{nom}'] = np.nan
            continue
        u, c = np.unique(np.round(v, 2), return_counts=True)
        res[f'moy_{nom}'] = float(v.mean())
        res[f'max_{nom}'] = float(v.max())
        res[f'maj_{nom}'] = float(u[c.argmax()])
    return res

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
    #lulc_2016 = lulc_2016.drop(['count', 'layer', 'path'], axis = 1) #suppression des colonnes non pertinentes
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

    #extraction du raster pour chaque polygone

    lignes = []
    for _, row in lulc_rand.iterrows():
        # pré-clip bbox : accélère clip
        s = stats_polygone(row.geometry, stack)
        if s is None:
            print(f'hors raster : id_u={row["id_u"]}')
            continue
        s['id_u'] = row['id_u']
        lignes.append(s)
    df_stats = pd.DataFrame(lignes)

    lulc_rand_final = lulc_rand.merge(df_stats, left_on='id_u', right_on='id_u', how='left')
    #récupèration du centroïd
    lulc_rand_final['centroid'] = lulc_rand_final.geometry.centroid
    lulc_sample = lulc_rand_final.copy()

    print(tabulate(lulc_sample[0:30], headers='keys'))

    return lulc_sample

lulc_sample = build_sample()


def model_rf() :

    """
    Paramétrage et lancement de la classification : modèle de classification supervisée Random Forest

    :return:
    """
    #Nom des données d'entrées : lulc_sample

    #Split des données
    X_train, X_test, y_train, y_test = train_test_split(lulc_sample['data'], lulc_sample['target'],test_size = 0.5, random_state = 42, stratify = lulc_sample['target'])

    #Random Forest
    rf_model = RandomForestClassifier(n_estimators = 200, max_depth = None, random_state = 42)
    rf_model.fit(X_train, y_train)

    # Accuracy
    predict_train_RF = rf_model.predict(X_train)
    predict_test_RF = rf_model.predict(X_test)
    acc_train_RF = accuracy_score(y_train, predict_train_RF)
    acc_test_RF = accuracy_score(y_test, predict_test_RF)

    print(f"Accuracy on the train dataset : {acc_train_RF:.4f}")
    print(f"Accuracy on the test dataset : {acc_test_RF:.4f}")

model_rf()

def model_XGBoost() :
    """
    Paramétrage et lancement de la classification : modèle de classification supervisée XGBoost

    :return:
    """
    #Nom des données d'entrées : lulc_sample

    #Split des données
    X_train, X_test, y_train, y_test = train_test_split(lulc_sample['data'], lulc_sample['target'], test_size = 0.5, random_state = 42, stratify=lulc_sample['target'])

    #XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    #Accuracy
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
    print(f"XGBoost — Exactitude : {xgb_acc:.4f}")

    predict_train = xgb_model.predict(X_train)
    predict_test = xgb_model.predict(X_test)
    acc_train = accuracy_score(y_train, predict_train)
    acc_test = accuracy_score(y_test, predict_test)

    print(f"Accuracy on the train dataset : {acc_train:.4f}")
    print(f"Accuracy on the test dataset : {acc_test:.4f}")

model_XGBoost()