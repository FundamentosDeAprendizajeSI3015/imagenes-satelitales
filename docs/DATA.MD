# Resumen de Datos y Preprocesamiento
## Proyecto: Detección de Deforestación en Antioquia

---

## 1. Fuente de Datos

- **Satélite:** Landsat 8 (LANDSAT/LC08/C02/T1_L2)
- **Proveedor:** USGS a través de Google Earth Engine (GEE)
- **Tipo:** Imágenes multiespectrales de reflectancia superficial (Surface Reflectance)
- **Colección:** Collection 2, Tier 1, mayor calidad disponible gratuitamente

---

## 2. Área de Estudio

- **Departamento:** Antioquia, Colombia
- **Bounding box:** [-77.5, 5.5, -74.0, 8.5] (longitud oeste, latitud norte)
- **Forma:** Rectangular (no sigue el límite político exacto del departamento)

---

## 3. Cobertura Temporal

- **Años exportados:** 2014, 2015, 2017, 2018, 2020, 2021, 2023
- **Intervalo aproximado:** 1.5 años entre imágenes
- **Ventana por imagen:** Año completo (enero–diciembre de cada año)
- **Total de imágenes:** 7

---

## 4. Preprocesamiento en GEE

### 4.1 Filtrado por Área
Cada colección anual fue filtrada espacialmente al bounding box de Antioquia, descartando escenas que no intersectan la región de interés.

### 4.2 Enmascaramiento de Nubes
Se aplicó una máscara de nubes pixel a pixel usando la banda **QA_PIXEL** de Landsat, que clasifica cada píxel según su condición atmosférica. Se eliminaron los píxeles clasificados como:

- Nubes (bit 3)
- Sombras de nubes (bit 4)
- Nubes dilatadas (bit 1)

Esto garantiza que el composite final solo incluya observaciones con cielo despejado. Los píxeles que permanecieron nublados durante todo el año quedan como NaN en el archivo final.

### 4.3 Composite Anual por Mediana
Con todos los píxeles nublados enmascarados, se calculó la **mediana** de todas las escenas disponibles durante el año completo. La mediana es robusta frente a valores atípicos residuales y produce un mosaico espacialmente completo aprovechando las distintas fechas de adquisición a lo largo del año.

### 4.4 Selección y Renombramiento de Bandas
Se seleccionaron 4 bandas espectrales y se renombraron para mayor claridad:

| Banda original | Nombre asignado | Región espectral |
|---|---|---|
| SR_B4 | Red | Rojo visible (~665 nm) |
| SR_B5 | NIR | Infrarrojo cercano (~865 nm) |
| SR_B6 | SWIR1 | Infrarrojo de onda corta 1 (~1610 nm) |
| SR_B7 | SWIR2 | Infrarrojo de onda corta 2 (~2200 nm) |

### 4.5 Corrección de Escala
Los valores digitales originales de Landsat se convirtieron a reflectancia superficial real aplicando los factores oficiales de escala de la Collection 2:

reflectancia = valor_digital × 0.0000275 + (−0.2)

Esto transforma los valores enteros almacenados por el satélite a valores físicos en el rango **0.0 – 1.0**, donde 1.0 representa reflectancia perfecta.

### 4.6 Cálculo de NDVI
Se calculó el Índice de Vegetación de Diferencia Normalizada (NDVI) como banda adicional derivada:
NDVI = (NIR − Red) / (NIR + Red)

Interpretación de valores:

| Rango NDVI | Tipo de cobertura |
|---|---|
| 0.6 – 1.0 | Vegetación densa y saludable (bosque) |
| 0.2 – 0.6 | Vegetación degradada o cultivos |
| 0.0 – 0.2 | Suelo desnudo o zonas urbanas |
| < 0.0 | Agua o superficies artificiales |

### 4.7 Conversión de Tipo de Dato
Todas las bandas fueron convertidas a **Float32** usando `.toFloat()` en GEE para garantizar consistencia de tipos entre las bandas espectrales (que salían como Float32) y el NDVI (que por defecto sale como Float64), evitando errores en la exportación.

### 4.8 Recorte al Área de Estudio
Cada imagen fue recortada al bounding box de Antioquia antes de exportar, eliminando datos fuera de la región de interés.
