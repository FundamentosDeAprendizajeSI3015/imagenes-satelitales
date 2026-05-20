# EDA Antioquia TIF

## Datos
- Archivos analizados: 7
- Años: 2014, 2015, 2017, 2018, 2020, 2021, 2023
- Shape por raster: 5574 x 6495 x 5
- CRS: EPSG:4326
- Bounds: [-76.500350, 5.499845, -72.999615, 8.504171]
- Dtype: float32

## Calidad / NaNs
- Mejor cobertura media: 2023 (99.47% pixeles validos por banda).
- Peor cobertura media: 2021 (97.58% pixeles validos por banda).
- `cloud_nan_frequency.png` muestra la frecuencia de pixeles sin dato a traves de los años.

## NDVI
- NDVI mediano minimo: 2015 (0.790).
- NDVI mediano maximo: 2017 (0.806).
- Revisa `ndvi_distribution_by_year.png` y `ndvi_thumbnails_by_year.png` para cambios temporales y posibles artefactos por nube.

## Proxy inicial de deforestacion T -> T+1
- Forest proxy en T: NDVI >= 0.65.
- Loss proxy en T+1: NDVI <= 0.45 y delta NDVI <= -0.2.
- Este target es exploratorio; debe calibrarse con verdad de terreno, Hansen/MapBiomas, etiquetas propias o validacion visual.

| year_t | year_tp1 | valid_overlap_pct | forest_area_ha_t | loss_proxy_area_ha | loss_proxy_pct_of_forest_t | delta_ndvi_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | 2015 | 96.983 | 11293222.320 | 95524.920 | 0.846 | -0.016 |
| 2015 | 2017 | 96.995 | 10787381.640 | 38252.520 | 0.355 | 0.020 |
| 2017 | 2018 | 98.218 | 11511401.760 | 26629.560 | 0.231 | -0.009 |
| 2018 | 2020 | 97.928 | 11312566.560 | 30920.760 | 0.273 | -0.007 |
| 2020 | 2021 | 96.770 | 10797337.800 | 53627.760 | 0.497 | 0.012 |
| 2021 | 2023 | 97.258 | 11358529.920 | 60517.800 | 0.533 | -0.003 |

## Archivos principales
- `metadata.csv`: metadatos y alineacion.
- `band_stats.csv`: estadisticas por banda y año.
- `change_pairs.csv`: cambios entre años consecutivos disponibles.
- `valid_year_count.npy`: numero de años validos por pixel.
- Figuras PNG: distribuciones, cobertura, NDVI y proxy de perdida.

## Siguientes paso
- Construir un dataset supervisado por pixel/patch con features de T y lags: bandas, NDVI, delta previo, frecuencia de nubes, vecinos espaciales y año.
- Definir target como `loss_proxy(T,T+1)` o reemplazarlo por etiquetas externas si existen.
- Separar train/test por tiempo y/o espacio, no aleatoriamente pixel a pixel, para evitar leakage espacial.
