# EDA NDVI Antioquia

Producto: MODIS MOD13Q1/MYD13Q1 v6.1, NDVI 16 dias, 250 m.
Metodo: promedio anual por pixel usando observaciones con confiabilidad buena o marginal.

## Observaciones usadas

|   year |   modis_items |
|-------:|--------------:|
|   2008 |            48 |
|   2013 |            48 |
|   2018 |            49 |
|   2023 |            46 |

## Resumen anual

|      year |   valid_pixels |   mean_ndvi |   median_ndvi |   std_ndvi |   p10_ndvi |   p25_ndvi |   p75_ndvi |   p90_ndvi |   low_vegetation_pct |   medium_vegetation_pct |   high_vegetation_pct |   mean_valid_observations |
|----------:|---------------:|------------:|--------------:|-----------:|-----------:|-----------:|-----------:|-----------:|---------------------:|------------------------:|----------------------:|--------------------------:|
| 2008.0000 |   1180131.0000 |      0.7762 |        0.7863 |     0.0802 |     0.6964 |     0.7406 |     0.8297 |     0.8564 |               0.0074 |                  0.1005 |                0.8921 |                   27.1201 |
| 2013.0000 |   1177628.0000 |      0.7852 |        0.7951 |     0.0821 |     0.7063 |     0.7499 |     0.8386 |     0.8663 |               0.0075 |                  0.0799 |                0.9126 |                   27.3655 |
| 2018.0000 |   1180397.0000 |      0.7855 |        0.7968 |     0.0820 |     0.7036 |     0.7506 |     0.8396 |     0.8668 |               0.0075 |                  0.0855 |                0.9069 |                   30.2758 |
| 2023.0000 |   1181456.0000 |      0.7867 |        0.7980 |     0.0838 |     0.7085 |     0.7545 |     0.8391 |     0.8669 |               0.0086 |                  0.0764 |                0.9150 |                   28.7785 |

## Analisis del cambio en intervalos de 5 años

El NDVI promedio de Antioquia se mantiene alto durante todo el periodo analizado. En 2008 el promedio fue 0.7762 y en 2023 fue 0.7867, una diferencia total de +0.0105. El cambio es pequeno en terminos absolutos, pero consistente con una cobertura vegetal regionalmente estable o ligeramente mas vigorosa en el promedio anual. La mediana tambien sube de 0.7863 a 0.7980, lo que indica que el incremento no depende solo de unos pocos pixeles extremos.

Entre 2008 y 2013 se observa el aumento mas claro: el NDVI medio sube +0.0090 y la proporcion de pixeles con vegetacion alta (`NDVI >= 0.7`) pasa de 89.21% a 91.26%. En el mismo intervalo, la vegetacion media baja de 10.05% a 7.99%. Esto sugiere que muchas zonas que estaban en rangos medios pasaron a rangos altos de verdor, posiblemente por recuperacion estacional promedio, cambios en humedad, regeneracion vegetal o diferencias interanuales en condiciones climaticas.

Entre 2013 y 2018 el cambio regional es casi plano: el NDVI medio apenas sube +0.0003. Sin embargo, la vegetacion alta baja de 91.26% a 90.69% y la vegetacion media sube de 7.99% a 8.55%. Esto sugiere estabilidad general, pero con una pequena redistribucion interna: algunas zonas pasan de valores altos a medios sin que el promedio departamental cambie demasiado.

Entre 2018 y 2023 hay un nuevo aumento leve: el NDVI medio sube +0.0012 y la vegetacion alta aumenta de 90.69% a 91.50%. Al mismo tiempo, la vegetacion baja tambien aumenta ligeramente de 0.75% a 0.86%. Esto significa que el promedio general mejora, pero no de forma completamente uniforme; hay sectores localizados donde el verdor bajo gana peso mientras el conjunto del departamento se mantiene muy verde.

## Inferencias principales

La senal departamental es de alta cobertura o vigor vegetal sostenido. Antioquia aparece dominada por valores de NDVI altos en los cuatro años, con mas del 89% de pixeles validos por encima de 0.7 en todos los cortes. Esto es coherente con un territorio donde grandes extensiones conservan vegetacion densa, cultivos permanentes, mosaicos agroforestales o coberturas con alta actividad fotosintetica.

El cambio mas importante ocurre entre 2008 y 2013. Despues de 2013, los cambios son menores y el sistema se comporta de manera estable. La lectura mas prudente es que no hay una degradacion generalizada visible a escala MODIS de 250 m para todo Antioquia entre 2008 y 2023. Si existieron perdidas locales, estas pueden quedar diluidas por el promedio departamental o compensadas por recuperacion vegetal en otras zonas.

La desviacion estandar sube de 0.0802 en 2008 a 0.0838 en 2023. Esto indica una ligera mayor heterogeneidad espacial: aunque el promedio sube, las diferencias internas entre zonas tambien aumentan. En otras palabras, Antioquia no cambia como una unidad homogenea; hay zonas con NDVI muy alto y otras con senales mas bajas o decrecientes.

## Clusters por cuadricula

|   cluster |     cells |   mean_lon |   mean_lat |   mean_ndvi_2008 |   mean_ndvi_2013 |   mean_ndvi_2018 |   mean_ndvi_2023 |   mean_change_2008_2023 |
|----------:|----------:|-----------:|-----------:|-----------------:|-----------------:|-----------------:|-----------------:|------------------------:|
|    0.0000 |  368.0000 |   -76.6204 |     8.0709 |           0.7624 |           0.7739 |           0.7672 |           0.7740 |                  0.0116 |
|    1.0000 | 1097.0000 |   -75.7465 |     7.0022 |           0.8261 |           0.8354 |           0.8361 |           0.8372 |                  0.0110 |
|    2.0000 |  326.0000 |   -75.4168 |     7.1585 |           0.6756 |           0.6862 |           0.6868 |           0.6865 |                  0.0109 |
|    3.0000 | 1341.0000 |   -75.2262 |     6.5421 |           0.7664 |           0.7748 |           0.7771 |           0.7777 |                  0.0113 |
|    4.0000 |   40.0000 |   -75.2201 |     6.8406 |           0.4982 |           0.4822 |           0.4852 |           0.4741 |                 -0.0241 |

## Lectura de clusters

Los clusters 0, 1, 2 y 3 muestran aumentos moderados entre 2008 y 2023, todos alrededor de +0.011 NDVI. Esto refuerza la idea de estabilidad o leve mejora en la mayor parte del departamento. El cluster 1 es el mas verde de forma persistente, con NDVI medio de 0.8261 en 2008 y 0.8372 en 2023.

El cluster 4 es el caso mas importante para revisar con detalle. Tiene pocas celdas, solo 40, pero presenta una caida de -0.0241 entre 2008 y 2023. Ademas, baja entre 2008 y 2013, recupera levemente en 2018 y vuelve a caer en 2023. Este patron puede corresponder a zonas urbanas, suelos expuestos, mineria, agua, nubes residuales, cambios de uso del suelo o coberturas agricolas con menor verdor anual. Para concluir deforestacion se deberia cruzar este cluster con `lossyear` de Hansen, imagenes Landsat/Sentinel o informacion local.

## Conclusiones

Antioquia mantiene un NDVI alto y estable en los cuatro años analizados. El promedio departamental no muestra una perdida generalizada de vigor vegetal; por el contrario, pasa de 0.7762 en 2008 a 0.7867 en 2023. La mejora mas fuerte se concentra entre 2008 y 2013, mientras que 2013-2018 y 2018-2023 son periodos de cambios pequenos.

La distribucion de NDVI se desplaza ligeramente hacia valores altos. La proporcion de vegetacion alta aumenta de 89.21% a 91.50% entre 2008 y 2023, mientras la vegetacion media disminuye. Esto sugiere que, a escala regional, el departamento se ve igual o mas verde en 2023 que en 2008.

La conclusion debe leerse con cautela porque NDVI no es una medicion directa de bosque. Un NDVI alto puede venir de bosque natural, cultivos, pastos muy activos o regeneracion; un NDVI bajo puede deberse a urbanizacion, agua, suelo desnudo, nubosidad residual o actividad minera. Para hablar especificamente de deforestacion, este EDA debe complementarse con la capa `lossyear` de Hansen o con clasificacion de coberturas.

## Archivos generados

- `ndvi_antioquia_2008.png`, `ndvi_antioquia_2013.png`, `ndvi_antioquia_2018.png`, `ndvi_antioquia_2023.png`
- `hist_ndvi_antioquia_2008.png`, `hist_ndvi_antioquia_2013.png`, `hist_ndvi_antioquia_2018.png`, `hist_ndvi_antioquia_2023.png`
- `annual_ndvi_summary.csv`
- `grid_clusters.csv`
- `cluster_summary.csv`
- `clusters_antioquia_ndvi_grid.png`
