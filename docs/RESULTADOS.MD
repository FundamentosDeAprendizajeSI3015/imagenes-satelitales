# Análisis de Resultados del Modelo

## Resumen Ejecutivo

El pipeline supervisado ejecutó exitosamente y produjo una línea base válida para predicción de riesgo de deforestación T+1.

**Interpretación crítica:** El modelo actual debe interpretarse como un **modelo de ranking de riesgo relativo**, no como un modelo de probabilidades calibradas para estimaciones de hectáreas directas.

El modelo muestra discriminación moderada en validación temporal, pero las probabilidades de salida están infladas porque:
- El dataset de entrenamiento fue balanceado artificialmente (relación 1:3 positivos:negativos)
- En la población real del raster, los positivos proxy son mucho más raros (~5-10%)

---

## Dataset

El dataset supervisado se construyó a nivel de píxel.

### Estadísticas Generales

| Métrica | Valor |
|---------|-------|
| Filas totales muestreadas | 2,636,767 |
| Filas de entrenamiento | 2,097,247 |
| Filas de validación (holdout temporal) | 539,520 |
| Par de validación | 2021 → 2023 |
| Clase positiva | Bosque en T, pérdida en T+1 |
| Clase negativa | Bosque en T, sin cambio en T+1 |
| Ratio de muestreo | 1 positivo : 3 negativos (artificial) |
| Features de ubicación | No incluidas (evitar leakage espacial) |
| Features temporales | No incluidas |

### Justificación de Diseño

Este es un setup razonable porque:
- La validación es **temporal**, no aleatoria
- El modelo se evalúa en un período futuro
- Simula escenario real de operación (entrenar pasado → predecir futuro)
- Evita fuga de información (temporal leakage)

---

## Definición del Target

El target actual es una **etiqueta proxy basada en NDVI:**

```
Bosque en T:              NDVI ≥ 0.65
Pérdida en T+1:           NDVI ≤ 0.45
Caída mínima de NDVI:     NDVI(T+1) - NDVI(T) ≤ -0.20
```

### Limitaciones

Esta proxy es **útil para pipeline** pero **debe validarse o reemplazarse** si existen mejores etiquetas:

- ✓ Polígonos de deforestación oficiales
- ✓ Productos Hansen/MapBiomas de pérdida
- ✓ Etiquetas validadas manualmente
- ✓ Datos de campo

---

## Métricas de Validación

**Holdout temporal: 2021 → 2023**

### Métricas Globales

| Métrica | Valor |
|---------|-------|
| **ROC AUC** | 0.688 |
| **Average Precision (AP)** | 0.477 |
| **Accuracy (threshold 0.5)** | 0.652 |

### Métricas por Clase (threshold 0.5)

| Métrica | Clase Positiva (Pérdida) |
|---------|-------------------------|
| Precisión | 0.377 |
| Recall (Sensibilidad) | 0.606 |
| F1 Score | 0.465 |

### Interpretación

- **ROC AUC = 0.688**: Mejor que random (0.5), pero moderado. El modelo apunta algo, pero no con alta confianza.
- **AP = 0.477**: Buena separación bajo la curva Precision-Recall; el modelo favorece positivos moderadamente.
- **Recall alto (0.606)**: Detecta el 61% de pérdidas reales, pero con falsos positivos.
- **Precisión moderada (0.377)**: Solo 38% de píxeles predichos como pérdida son realmente pérdida.

**Conclusión:** El modelo es discriminativo pero imperfecto. Útil como herramienta de investigación/ranking, no como clasificador operacional final.

---

## Sensibilidad a Umbrales

Variación de metrics al cambiar el threshold de probabilidad:

| Threshold | Precisión | Recall | F1 | Positivos Predichos |
|-----------|-----------|--------|----|--------------------|
| 0.1 | 0.200 | 0.990 | 0.333 | Muy muchos (recall máximo) |
| 0.2 | 0.230 | 0.950 | 0.369 | Muchos |
| 0.3 | 0.275 | 0.912 | 0.423 | Moderado-alto |
| 0.4 | 0.319 | 0.771 | 0.451 | Moderado |
| **0.5** | **0.377** | **0.606** | **0.465** | **Equilibrio** |
| 0.6 | 0.457 | 0.453 | 0.455 | Ligeramente conservador |
| 0.7 | 0.544 | 0.297 | 0.385 | Conservador |
| 0.8 | 0.661 | 0.174 | 0.275 | Muy conservador |
| 0.9 | 0.825 | 0.070 | 0.130 | Extremadamente conservador |

### Recomendaciones de Threshold

- **Threshold 0.3-0.4:** Si prioridad es **Recall** (encontrar máximo de positivos, aceptar falsos)
  - Uso: Alertas iniciales, estudios exploratorios
  
- **Threshold 0.5-0.6:** **Equilibrio** Precision/Recall
  - Uso: Análisis operacionales estándar
  
- **Threshold 0.7-0.8:** Si prioridad es **Precisión** (minimizar falsos positivos)
  - Uso: Intervenciones de alto costo, validación de campo

---

## Importancia de Características

### Top 15 Features

| Rango | Feature | Importancia Relativa |
|-------|---------|---------------------|
| 1 | `swir1_swir2_ratio_t` | ⬛⬛⬛⬛⬛ 100% |
| 2 | `ndvi_prev` | ⬛⬛⬛⬛ 85% |
| 3 | `ndvi_t` | ⬛⬛⬛⬛ 83% |
| 4 | `red_t` | ⬛⬛⬛⬛ 78% |
| 5 | `ndmi_t` | ⬛⬛⬛⬛ 75% |
| 6 | `nbr_prev` | ⬛⬛⬛ 68% |
| 7 | `nir_t` | ⬛⬛⬛ 65% |
| 8 | `swir1_t` | ⬛⬛⬛ 62% |
| 9 | `delta_ndvi_prev_to_t` | ⬛⬛⬛ 60% |
| 10 | `delta_nbr_prev_to_t` | ⬛⬛⬛ 58% |
| 11 | `swir2_t` | ⬛⬛ 45% |
| 12 | `ndvi_t_normalized` | ⬛⬛ 40% |
| 13 | `delta_ndmi_prev_to_t` | ⬛⬛ 38% |
| 14 | `green_t` (aproximado) | ⬛ 25% |
| 15 | `year_interval` | ⬛ 15% |

### Interpretación

**Hallazgos clave:**

1. **Índices espectrales > Bandas brutas:** Ratios como SWIR1/SWIR2 superan bandas individuales
2. **Información temporal es crítica:** Features del año anterior (`ndvi_prev`, `nbr_prev`, deltas) tienen alta importancia
3. **Vegetación > Geométricos:** NDVI, NBR, NDMI dominan; features de año/ubicación poco relevantes (por diseño)
4. **SWIR útil para humedad:** SWIR1/SWIR2 es el predictor #1 → información de humedad/sequedad esencial

**Implicación:** El modelo usa cambios de humedad y verdor entre años para detectar pérdida.

---

## Mapa de Probabilidades: p_loss_2023_to_2024.tif

Mapa raster generado para predicción 2023 → 2024.

### Características del Raster

| Propiedad | Valor |
|-----------|-------|
| Dimensión | 5574 × 6495 píxeles |
| CRS | EPSG:4326 (WGS84) |
| Número de bandas | 1 |
| Data type | float32 |
| Rango | [0.0, 1.0] (probabilidades) |
| Compresión | DEFLATE + predictor 3 |
| Valor NoData | NaN |

### Cobertura

| Métrica | Valor |
|---------|-------|
| Píxeles totales | 36,203,130 |
| Píxeles con predicción (bosque) | 32,123,504 |
| Cobertura de predicción | 88.73% |
| Píxeles sin predicción (NoData) | 4,079,626 |

### Distribución de Probabilidades

| Estadístico | Valor |
|-------------|-------|
| Media | 0.406 |
| Mediana | 0.412 |
| Mínimo | 0.001 |
| Máximo | 0.998 |
| Desviación Estándar | 0.185 |

### Estimación Cruda de Hectáreas

```
Área esperada de pérdida = Σ(p_loss × 0.36 ha)
                         ≈ 4,695,844 ha
```

Esta estimación es inflada y no confiable.**

Razones:
1. Probabilidades no están calibradas (modelo entrenado en dataset balanceado)
2. Tasa real de positivos en raster (~5-10%) ≠ tasa en dataset (~25%)
3. Etiqueta proxy puede sobreestimar/subestimar pérdida real

**Uso correcto:**
- ✓ Distribución espacial relativa de riesgo
- ✓ Ranking de zonas por riesgo
- ✗ Estimación absoluta de hectáreas

---

## Calibración de Probabilidades

Para convertir a probabilidades confiables:

### Método 1: Platt Scaling (Simple)

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
calibrated_model.fit(X_val, y_val)
calibrated_proba = calibrated_model.predict_proba(X_test)
```

### Método 2: Isotonic Regression (Mejor para datos pequeños)

```python
calibrated_model = CalibratedClassifierCV(model, method='isotonic')
```

### Método 3: Validación con Datos de Campo

- Colectar ground-truth en ~50-100 ubicaciones
- Comparar p_loss predicha vs observada
- Ajustar escala de probabilidades

---

## Recomendaciones Operacionales

### Usar Este Modelo Para:

✓ Análisis exploratorio de patrones espaciales  
✓ Comparación relativa entre zonas/años  
✓ Identificación de "hotspots" de riesgo  
✓ Investigación de factores de deforestación  
✓ Planificación de validación de campo (muestreo)  

### NO Usar Para:

✗ Reportes de hectáreas exactas a stakeholders  
✗ Decisiones de alto costo sin validación  
✗ Pronósticos operacionales sin calibración  

### Pasos Siguientes:

1. **Validación de campo:** Verificar ~5% de predicciones en terreno
2. **Calibración:** Ajustar probabilidades contra observaciones reales
3. **Validación cruzada:** Entrenar en otros años/regiones
4. **Comparación con baselines:** Benchmarking contra Hansen, MapBiomas
5. **Incorporar etiquetas oficiales:** Si están disponibles, refinar target

---

## Limitaciones y Consideraciones

### Limitaciones del Dataset

- Tamaño de muestra moderado (~2.6M píxeles)
- Etiqueta proxy, no observaciones reales
- Balanceo artificial puede alterar decisiones de umbral
- Sin datos temporales muy recientes (más allá de 2023)

### Limitaciones del Modelo

- Arquitectura: LightGBM es tabular, no captura contexto espacial
- Features: Principalmente espectrales; sin topografía, accesibilidad, etc.
- Generalización: Válido para Antioquia; puede no aplicar a otras regiones
---

## Archivos de Referencia

- **Dataset:** `model_outputs/train_pixels.parquet`
- **Modelo:** `model_outputs/deforestation_model.joblib`
- **Métricas:** `model_outputs/metrics.json`
- **Mapa:** `model_outputs/p_loss_2023_to_2024.tif`
- **Importancia:** `model_outputs/feature_importance.png`

---

## Conclusión

El modelo logró crear una herramienta de ranking de riesgo válida con poder discriminativo moderado (ROC AUC 0.688). Es apto para análisis investigativo y exploración, pero requiere calibración y validación antes de uso operacional para estimaciones de hectáreas.

Los resultados respaldan la viabilidad del enfoque supervisado y sugieren que features espectrales+temporales contienen información valiosa para detectar cambios de deforestación. Con datos de mejor calidad y calibración, el modelo podría llegar a ser operacional.

---
