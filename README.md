# Predicción de Deforestación T+1 con Aprendizaje Profundo

**Proyecto Final - Fundamentos de Aprendizaje Profundo**  
*Semestre VII - EAFIT*

## Descripción General

Este proyecto implementa un pipeline completo de aprendizaje automático supervisado para predecir cambios de deforestación pixel por pixel en imágenes satelitales multiespectrales de Antioquia. 

**Flujo:**
1. **Análisis Exploratorio (EDA)** — Estadísticas de bandas espectrales y validación temporal
2. **Análisis de Clustering** — PCA + K-means para caracterizar píxeles sin etiquetas
3. **Construcción de Dataset Supervisado** — Generación de muestras de entrenamiento T → T+1
4. **Entrenamiento del Modelo** — LightGBM con validación temporal (holdout por año)
5. **Predicción Espacial** — Mapas de probabilidad de pérdida forestal

---

## Requisitos

### Software
- Python 3.8+
- GDAL/rasterio (para manejo de GeoTIFF)
- scikit-learn, pandas, numpy
- LightGBM (opcional, usará HistGradientBoosting como fallback)

### Datos
- Rasters multiespectrales: `antioquia_YYYY.tif` (5 bandas: R, NIR, SWIR1, SWIR2, NDVI)
- Ubicación: `raw_data/`

### Instalación

```bash
# Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit==1.28.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
rasterio==1.3.7
joblib==1.3.1
pyarrow==12.0.1
scikit-learn==1.3.0
scipy==1.11.2
lightgbm==4.0.0
```

---

## Estructura del Proyecto

```
ProyectoFinal/
├── src/
│   ├── data/
│   │   └── build_supervised_dataset.py     # Construcción de dataset
│   ├── analysis/
│   │   ├── clustering_pca_analysis.py      # PCA + K-means
│   │   ├── compare_chunk_clustering.py     # Comparación NDVI vs clustering
│   │   ├── spatial_clustering_map.py       # Mapeo espacial de clusters
│   │   └── clustering_analysis_unified.py  # Pipeline unificado (RECOMENDADO)
│   ├── models/
│   │   └── train_deforestation_model.py    # Entrenamiento del modelo
│   └── prediction/
│       ├── predict_deforestation_map.py    # Predicción en raster
│       ├── visualize_predictions.py        # Dashboard Streamlit
│       └── test_visualization.py           # Pruebas visualización
├── raw_data/
│   └── antioquia_*.tif
├── outputs/
│   ├── clustering/
│   │   ├── pca_clusters_2d.png
│   │   ├── kmeans_metrics.png
│   │   ├── kmeans_metrics.csv
│   │   └── ndvi_clustering_comparison_*.png
│   ├── eda/
│   ├── models/
│   └── results/
├── model_outputs/
│   ├── train_pixels.parquet
│   ├── train_pixels.manifest.json
│   ├── deforestation_model.joblib
│   ├── metrics.json
│   ├── threshold_metrics.csv
│   ├── features.json
│   └── p_loss_*.tif
├── docs/
│   ├── DATA.MD
│   ├── WORKFLOW.MD
│   ├── PIPELINE.MD
│   └── RESULTADOS.MD
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pasos de Ejecución

### 1. Análisis Exploratorio de Datos (EDA)

```bash
python eda_outputs/eda_report.py
```

**Salida:**
- `eda_outputs/band_stats.csv` — Estadísticas por banda
- `eda_outputs/metadata.csv` — Información temporal y geoespacial
- `eda_outputs/eda_report.md` — Reporte completo

### 2. Análisis de Clustering (RECOMENDADO: Script Unificado)

```bash
python src/analysis/clustering_analysis_unified.py \
  --data-dir raw_data \
  --year 2023 \
  --out-dir outputs/clustering \
  --sample-size 50000 \
  --num-chunks 6
```

**Salida:**
- `outputs/clustering/pca_clusters_2d.png` — Visualización PCA
- `outputs/clustering/kmeans_metrics.png` — Métricas de K-means
- `outputs/clustering/ndvi_clustering_comparison_2023.png` — Comparación de chunks
- `outputs/clustering/kmeans_metrics.csv` — Métricas en CSV
- `outputs/clustering/chunk_comparison_stats_2023.json` — Estadísticas detalladas

### 3. Construcción del Dataset Supervisado

```bash
python src/data/build_supervised_dataset.py \
  --data-dir raw_data \
  --out model_outputs/train_pixels.parquet \
  --negative-ratio 3 \
  --forest-ndvi-threshold 0.65 \
  --loss-ndvi-threshold 0.45 \
  --min-ndvi-drop -0.20
```

**Parámetros:**
- `--negative-ratio 3`: Relación 1 positivo : 3 negativos
- `--forest-ndvi-threshold 0.65`: Umbral mínimo para considerar "bosque"
- `--loss-ndvi-threshold 0.45`: Umbral máximo para considerar "pérdida"
- `--min-ndvi-drop -0.20`: Caída mínima de NDVI para confirmar pérdida

**Salida:**
- `model_outputs/train_pixels.parquet` — Dataset Parquet (~3M píxeles)
- `model_outputs/train_pixels.manifest.json` — Metadatos del dataset

### 4. Entrenamiento del Modelo

```bash
python src/models/train_deforestation_model.py \
  --dataset model_outputs/train_pixels.parquet \
  --out-dir model_outputs \
  --holdout-year-t 2021 \
  --n-estimators 600 \
  --learning-rate 0.04
```

**Parámetros:**
- `--holdout-year-t 2021`: Validación temporal (datos de 2021→2023 como test)
- `--n-estimators 600`: Número de árboles
- `--learning-rate 0.04`: Tasa de aprendizaje

**Salida:**
- `model_outputs/deforestation_model.joblib` — Modelo entrenado
- `model_outputs/metrics.json` — ROC AUC, AP, reportes de clasificación
- `model_outputs/threshold_metrics.csv` — Métricas a diferentes umbrales
- `model_outputs/feature_importance.png` — Importancia de características

### 5. Predicción en Raster Completo

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

**Salida:**
- `model_outputs/p_loss_2023_to_2024.tif` — Mapa de probabilidad de pérdida

### 6. Visualización Interactiva (Streamlit)

```bash
streamlit run src/prediction/visualize_predictions.py
```

Abre dashboard interactivo en `http://localhost:8501`

---

## Definición del Target (Etiqueta Supervisada)

El modelo usa una **etiqueta proxy** basada en NDVI:

```
Bosque en T:          NDVI ≥ 0.65
Pérdida en T+1:       NDVI ≤ 0.45
Caída mínima:         ∆NDVI ≤ -0.20
```

**Nota:** Si dispone de etiquetas oficiales (polígonos de deforestación, Hansen/MapBiomas), reemplace esta proxy.

---

## Resultados Esperados

### Métricas de Validación (Temporal Holdout 2021→2023)

| Métrica | Valor |
|---------|-------|
| ROC AUC | 0.688 |
| Average Precision | 0.477 |
| Precisión (threshold 0.5) | 0.377 |
| Recall (threshold 0.5) | 0.606 |
| F1 (threshold 0.5) | 0.465 |

### Características Más Importantes

1. Ratio SWIR1/SWIR2
2. NDVI anterior
3. NDVI actual
4. Band Roja
5. NDMI (Índice de Humedad)

### Interpretación

- **ROC AUC = 0.688**: Modelo discriminativo moderado (mejor que random 0.5)
- **AP = 0.477**: Buena separación de clases positivas
- **Probabilidades infladas**: Dataset balanceado artificialmente (25% positivos); en raster real ~5-10% positivos
- **Use as relative ranking**: Mapa útil como *ranking de riesgo relativo*, no como probabilidades calibradas

---

## Guía de Documentación

### 📄 [DATA.MD](docs/DATA.MD)
Descripción de fuentes de datos, bandas espectrales, índices derivados, y validación de alineación temporal.

### 📄 [WORKFLOW.MD](docs/WORKFLOW.MD)
Overview de pasos de preprocessing y características generadas.

### 📄 [PIPELINE.MD](docs/PIPELINE.MD)
Guía técnica completa: definición del target, comandos bash, interpretación de output.

### 📄 [RESULTADOS.MD](docs/RESULTADOS.MD)
Análisis de resultados del modelo, métricas de validación, recomendaciones operacionales.

---

## Tips Prácticos

### Ejecución Rápida (Prueba)

```bash
# Clustering en muestra pequeña
python src/analysis/clustering_analysis_unified.py \
  --sample-size 10000 --num-chunks 3

# Dataset con máximo de filas
python src/data/build_supervised_dataset.py \
  --max-rows 500000

# Entrenamiento con pocos árboles
python src/models/train_deforestation_model.py \
  --max-rows 200000 --n-estimators 100
```

### Depuración

```bash
# Ver estructura de raster
python -c "import rasterio; src=rasterio.open('raw_data/antioquia_2023.tif'); print(src.shape, src.crs, src.bounds)"

# Inspeccionar Parquet
python -c "import pandas as pd; df=pd.read_parquet('model_outputs/train_pixels.parquet', columns=['target_loss','ndvi_t']); print(df.head(), df['target_loss'].value_counts())"
```

### Ajuste de Parámetros

**Para mayor Recall (encontrar más positivos):**
```bash
python src/models/train_deforestation_model.py \
  --learning-rate 0.08 --n-estimators 800
```

**Para mayor Precisión (menos falsos positivos):**
```bash
# En visualización, usar threshold más alto (0.6-0.7)
```

---

## Solución de Problemas

| Error | Solución |
|-------|----------|
| `FileNotFoundError: antioquia_*.tif` | Verificar que los rasters estén en `raw_data/` |
| `ImportError: No module named 'rasterio'` | `pip install rasterio` (puede requerir GDAL del sistema) |
| `MemoryError` durante PCA | Reducir `--sample-size` o procesar en chunks |
| Predicción lenta | Usar `--max-rows` en entrenamiento para modelo más pequeño |
| Métricas muy bajas | Validar definición del target proxy (NDVI thresholds) |

---

## Contribuciones y Mejoras Futuras

- [ ] Integración con etiquetas oficiales (Hansen, MapBiomas)
- [ ] Calibración de probabilidades (Platt scaling, isotonic regression)
- [ ] Modelos con redes neuronales (CNN para contexto espacial)
- [ ] Análisis de cambios multi-temporales (series de tiempo)
- [ ] Validación con datos de campo
- [ ] API REST para predicción en tiempo real

---

## Licencia

Proyecto académico - Todos los derechos reservados.

---

## Contacto

**Autor:** Sebastián (EAFIT)  
**Semestre:** VII  
**Curso:** Fundamentos de Aprendizaje Profundo  
**Fecha:** Mayo 2026

Para preguntas sobre el código o metodología, revisar la [documentación técnica](docs/).

---

## Resumen Rápido de Comandos

```bash
# 1. Análisis
python src/analysis/clustering_analysis_unified.py --year 2023

# 2. Dataset
python src/data/build_supervised_dataset.py

# 3. Entrenamiento
python src/models/train_deforestation_model.py --holdout-year-t 2021

# 4. Predicción
python src/prediction/predict_deforestation_map.py --year-t 2023

# 5. Visualización
streamlit run src/prediction/visualize_predictions.py
```
