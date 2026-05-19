# Flujo de Trabajo: Predicción Supervisada T+1 de Deforestación

## Target (Variable Objetivo)

El modelo se entrena a nivel de píxel, **no a nivel de hectárea**.

Para un par temporal T → T+1:

- **Población candidata:** píxeles que parecen bosque en T
- **Clase positiva (target_loss = 1):** Bosque-like en T, luego se convierte en loss-like en T+1
- **Clase negativa (target_loss = 0):** Bosque-like en T, se mantiene sin pérdida en T+1

### Etiqueta Proxy Predeterminada

```
Bosque en T:              NDVI ≥ 0.65
Pérdida en T+1:           NDVI ≤ 0.45
Caída mínima de NDVI:     NDVI(T+1) - NDVI(T) ≤ -0.20
```

**Nota importante:** Esta es una etiqueta proxy. Si existen etiquetas oficiales como:
- Polígonos de deforestación validados
- Rasters de pérdida Hansen o MapBiomas
- Etiquetas validadas manualmente

Entonces **use esas como target en lugar de la proxy NDVI**.

### Agregación a Hectáreas

Las hectáreas se calculan como agregación DESPUÉS de la predicción:

```
hectáreas_esperadas_pérdida = Σ(probabilidad_pérdida_píxel × área_píxel_ha)
```

Para estos píxeles de 60 metros, la aproximación de trabajo es:

```
área_píxel_ha = 0.36 ha
```

---

## 1. Construcción del Dataset

### Comando

```bash
python src/data/build_supervised_dataset.py \
  --data-dir raw_data \
  --out model_outputs/train_pixels.parquet \
  --negative-ratio 3 \
  --forest-ndvi-threshold 0.65 \
  --loss-ndvi-threshold 0.45 \
  --min-ndvi-drop -0.20
```

### Parámetros

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `--data-dir` | `.` | Directorio con archivos `antioquia_*.tif` |
| `--out` | `model_outputs/train_pixels.parquet` | Ruta de salida del dataset |
| `--negative-ratio` | `3.0` | Relación 1 positivo : N negativos (muestreo balanceado) |
| `--forest-ndvi-threshold` | `0.65` | NDVI mínimo para considerar "bosque" |
| `--loss-ndvi-threshold` | `0.45` | NDVI máximo para considerar "pérdida" |
| `--min-ndvi-drop` | `-0.20` | Caída mínima de NDVI entre T y T+1 |
| `--include-coords` | `False` | Incluir coordenadas (lon/lat) como características |
| `--seed` | `42` | Semilla para reproducibilidad |

### Características Incluidas

El dataset incluye:

- **Bandas espectrales en T:** red_t, nir_t, swir1_t, swir2_t
- **Índices derivados en T:** ndvi_t, ndmi_t, nbr_t, swir1_swir2_ratio_t
- **Deltas temporales (si hay año previo):** delta_ndvi, delta_ndmi, delta_nbr
- **Metadatos:** year_t, year_tp1, pair_index, row, col, pixel_area_ha
- **Target:** target_loss (0 o 1)

### Salida

```
model_outputs/
├── train_pixels.parquet              # Dataset (~3M píxeles)
└── train_pixels.manifest.json        # Metadatos del dataset
```

---

## 2. Entrenamiento del Modelo

### Comando

```bash
python src/models/train_deforestation_model.py \
  --dataset model_outputs/train_pixels.parquet \
  --out-dir model_outputs \
  --holdout-year-t 2021 \
  --seed 42 \
  --n-estimators 600 \
  --learning-rate 0.04
```

### Parámetros

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `--dataset` | `model_outputs/train_pixels.parquet` | Ruta al dataset de entrenamiento |
| `--out-dir` | `model_outputs` | Directorio para guardar modelo y métricas |
| `--holdout-year-t` | `None` (último año) | Año T para validación temporal |
| `--use-location` | `False` | Incluir features espaciales (row, col, lon, lat) |
| `--use-year` | `False` | Incluir features temporales (year_t, prev_year) |
| `--max-rows` | `None` | Limitar filas para pruebas rápidas |
| `--n-estimators` | `600` | Número de árboles/iteraciones |
| `--learning-rate` | `0.04` | Tasa de aprendizaje |
| `--seed` | `42` | Semilla para reproducibilidad |

### Validación Temporal Recomendada

Para un dataset con pares 2004→2006, 2006→2008, ..., 2021→2023:

**Mejor práctica:** Usar el par más reciente como holdout:
```bash
--holdout-year-t 2021  # Entrena con pares anteriores, valida en 2021→2023
```

Esto simula validación en **datos del futuro**, evitando fuga de datos temporal.

### Prueba Rápida en HPC/Dev Node

```bash
python src/models/train_deforestation_model.py \
  --dataset model_outputs/train_pixels.parquet \
  --out-dir model_outputs_smoke \
  --holdout-year-t 2021 \
  --max-rows 200000 \
  --n-estimators 100
```

### Salida

```
model_outputs/
├── deforestation_model.joblib       # Modelo entrenado (Pipeline + LightGBM/HistGradient)
├── features.json                    # Lista de características usadas
├── metrics.json                     # ROC AUC, AP, classification_report
├── threshold_metrics.csv            # Métricas a diferentes umbrales
├── feature_importance.png           # Gráfico de importancia
├── feature_importance.csv           # Importancia en CSV
├── pr_curve.png                     # Curva Precision-Recall
└── predictions_test.parquet         # Predicciones en conjunto de prueba
```

### Interpretación de Métricas

- **ROC AUC:** Capacidad del modelo para diferenciar positivos de negativos. Rango [0, 1]; 0.5 = random.
- **AP (Average Precision):** Integral bajo la curva Precision-Recall. Sensible a desbalance.
- **Threshold:** Punto de corte para convertir probabilidad continua a clasificación binaria.

---

## 3. Predicción en Raster Completo

### Comando

```bash
python src/prediction/predict_deforestation_map.py \
  --data-dir raw_data \
  --year-t 2023 \
  --prev-year 2021 \
  --interval-years 1 \
  --model model_outputs/deforestation_model.joblib \
  --features model_outputs/features.json \
  --out model_outputs/p_loss_2023_to_2024.tif
```

### Parámetros

| Parámetro | Descripción |
|-----------|-------------|
| `--data-dir` | Directorio con rasters `antioquia_*.tif` |
| `--year-t` | Año T (base) para predicción |
| `--prev-year` | Año anterior (para features temporales) |
| `--interval-years` | Intervalo hacia T+1 |
| `--model` | Ruta al modelo entrenado |
| `--features` | Ruta al archivo features.json |
| `--out` | Ruta de salida del raster de predicción |
| `--forest-ndvi-threshold` | NDVI mínimo para aplicar modelo |

### Output

```
model_outputs/p_loss_2023_to_2024.tif
```

**Características:**
- **1 banda:** Probabilidad de pérdida (float32, rango [0,1])
- **Píxeles con predicción:** Aquellos con NDVI ≥ threshold en T
- **Píxeles sin predicción:** NaN (no son bosque o inválidos)
- **Compresión:** DEFLATE con predictor=3

### Interpretación

**NO interprete directamente como probabilidades calibradas.** El modelo fue entrenado en dataset balanceado artificialmente (~25% positivos), mientras que el raster real tiene mucho menos positivos (~5-10%).

**Use como:**
- Mapa de **ranking de riesgo relativo**
- Input para análisis secundarios (umbralización, agregación)
- Comparación de zonas/épocas

**Para probabilidades calibradas:**
- Aplicar Platt scaling o isotonic regression
- Validar contra datos de campo

---

## Flujo Completo (Resumen)

```bash
# 1. Análisis exploratorio
python src/analysis/clustering_analysis_unified.py --year 2023

# 2. Construir dataset supervisado
python src/data/build_supervised_dataset.py

# 3. Entrenar modelo
python src/models/train_deforestation_model.py --holdout-year-t 2021

# 4. Predecir en raster
python src/prediction/predict_deforestation_map.py --year-t 2023

# 5. Visualizar (opcional)
streamlit run src/prediction/visualize_predictions.py
```

---

## Consideraciones Avanzadas

### Reducción de Fuga de Datos Espacial

Por defecto, las características **no incluyen coordenadas** (`--use-location` = False) para minimizar leakage espacial (el modelo no debería memorizar localizaciones).

Si necesita usar ubicación explícitamente, justifique por qué y documente el riesgo.

### Balanceo de Clases

El parámetro `--negative-ratio 3` sampled artificialmente para conseguir 1 positivo : 3 negativos.

Si la población real es 1:10 o 1:20, la relación apropiada depende de:
- Disponibilidad de datos positivos
- Necesidad de recall vs precisión en operaciones
- Capacidad computacional

### Validación Cruzada

El pipeline usa **temporal holdout** en lugar de k-fold aleatorio, porque:
- Simula escenario real: entrenar en pasado, predecir en futuro
- Evita fuga temporal
- Más conservador y realista

---

## Archivos de Salida Clave

| Archivo | Propósito |
|---------|-----------|
| `train_pixels.parquet` | Dataset de entrenamiento |
| `deforestation_model.joblib` | Modelo entrenado (reproducible) |
| `features.json` | Lista exacta de características usadas |
| `metrics.json` | Métricas de validación completas |
| `threshold_metrics.csv` | Análisis de sensibilidad a umbrales |
| `p_loss_YYYY_to_YYYY.tif` | Mapa raster de probabilidades |

---

## Reproducibilidad

Para garantizar reproducibilidad:

```bash
# Use seeds consistentes
python src/data/build_supervised_dataset.py --seed 42
python src/models/train_deforestation_model.py --seed 42
```

Documente:
- Versión de Python y librerías (en `requirements.txt`)
- Parámetros exactos usados
- Fecha de ejecución
- Maquina (CPU/GPU disponible)

---
