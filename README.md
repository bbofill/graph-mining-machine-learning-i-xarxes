# Movie Recommender Evolution: De POO Básica a Graph Machine Learning

Este repositorio documenta la evolución técnica y metodológica de un sistema de recomendación cinematográfico. El proyecto original nació durante mi primer curso universitario en la asignatura de **Programación Avanzada**, donde el objetivo principal era asimilar los fundamentos de la programación orientada a objetos (POO), abstracciones básicas y algoritmos heurísticos de filtrado. 

Tiempo después, y tras cursar materias avanzadas y adquirir conocimiento por propia voluntad, decidí retomar aquel código inicial. El objetivo de este rediseño es doble: medir mi propio progreso técnico y analizar empíricamente si modelar las relaciones entre usuarios y películas mediante estructuras de red aporta una ventaja real sobre los métodos clásicos.

---

## Preguntas de Investigación

El desarrollo de la segunda versión se plantea como un experimento controlado para responder a dos cuestiones prácticas:

* **RQ1 — Rendimiento Estructural:** ¿Superan los métodos basados en grafos (como *Personalized PageRank* o *LightGCN*) a los baselines clásicos de filtrado colaborativo matricial en métricas estrictas de ranking (ej. **NDCG@10**)?
* **RQ2 — Enriquecimiento Semántico:** ¿Mejora la precisión de la recomendación al evolucionar de un grafo puramente bipartito `Usuario–Película` a un grafo heterogéneo `Usuario–Película–Género` bajo el mismo protocolo experimental?

---

## Organización del repositorio

El proyecto está estructurado en dos carpetas independientes para poder contrastar el código base original con la solución final basada en grafos:

### `punto_de_partida_legacy/` (Versión inicial de 1º)
Código original de la asignatura de Programación Avanzada:
* `main_app.py`: Interfaz por terminal con split 80/20 secuencial.
* `recommendation_algorithms.py`: Lógica inicial de popularidad, KNN usuario-usuario e índice TF-IDF.
* `models_items.py` y `data_management.py`: Clases base (`Movie`, `Book`) y parseo manual de CSVs.
* `evaluation_metrics.py`: Evaluación clásica centrada en predecir notas (MAE y RMSE).

### `sistema_final_graph_mining/` (Proyecto final)
Rediseño completo enfocado a minería de grafos y evaluación de ranking:
* `main.py`: Ejecución principal (suite de evaluación masiva o modo interactivo).
* `src/data/`: Descarga y caché de MovieLens 1M, junto con el split temporal estricto por usuario.
* `src/graph/`: Generación de grafos bipartitos y tripartitos (`NetworkX`) y matrices laplacianas dispersas.
* `src/models/`: Modelos de grafo (Personalized PageRank, LightGCN/GNN) y baselines optimizados (SVD+SGD, KNN matricial).
* `src/evaluation/`: Protocolo experimental formal (NDCG@10, Precision, Recall y análisis de sensibilidad).
* `src/visualization/`: Exportación a PyVis (HTML interactivo), Gephi (.gexf) y proyecciones t-SNE.
* `results/`: Carpeta de salida automática para tablas de métricas, trazas y gráficas.
