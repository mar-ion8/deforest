'''
1. installer la bibliothèque earthengine dans le terminal :
dans le terminal : pip install earthengine-api

2. configurer l'authentification Google Cloud/Earth Engine :
dans le terminal : earthengine authenticate
créer un nouveau projet et activer l'API earthengine et bien indiquer un projet non-commerciale
'''
import os
import ee
import rasterio
from rasterio.warp import transform_bounds

ee.Initialize(project='deforestation-515009') # mettre l'ID du projet Google cloud

# 1. Charger l'emprise de ton image locale (GeoTIFF)
local_image_path = (  # Remplace par le chemin réel de ton fichier
    "C:/Users/gines/Desktop/Deforrestation/0_data_porto-velho/LC08_L2SP_232066_20160803_20200906_02_T1/LC08_L2SP_232066_20160803_20200906_02_T1_SR_B1.TIF"
)
with rasterio.open(local_image_path) as src:
  left, bottom, right, top = transform_bounds(
      src.crs, 'EPSG:4326', *src.bounds
  )

# 2. Définir le découpage en grille (ex: 3 colonnes et 3 lignes)
cols = 3
rows = 3

dx = (right - left) / cols
dy = (top - bottom) / rows

# 3. Charger MapBiomas Brésil 2016
mapbiomas_img = (
    ee.ImageCollection('projects/mapbiomas-public/assets/brazil/lulc/v1')
    .filter(ee.Filter.eq('year', 2016))
    .first()
    .select('classification')
)

# 4. Boucle pour générer et lancer une tâche d'export par tuile
index = 0
for i in range(cols):
  for j in range(rows):
    tile_xmin = left + i * dx
    tile_xmax = left + (i + 1) * dx
    tile_ymin = bottom + j * dy
    tile_ymax = bottom + (j + 1) * dy

    tile_geom = ee.Geometry.Rectangle(
        [tile_xmin, tile_ymin, tile_xmax, tile_ymax]
    )

    vector_data = mapbiomas_img.clip(tile_geom).reduceToVectors(
        geometry=tile_geom,
        scale=30,
        geometryType='polygon',
        eightConnected=False,
        labelProperty='class',
        maxPixels=1e9,
    )

    task = ee.batch.Export.table.toDrive(
        collection=vector_data,
        description=f'PortoVelho_Tile_{index}_2016',
        fileFormat='GeoJSON',
        folder='EarthEngine_Exports',
    )
    task.start()
    print(f'Tâche {index} lancée pour la tuile ({i}, {j}).')
    index += 1
