| Modelo | NDCG@10 (±IC 95%) | Precision@10 | Recall@10 |
| :--- | :---: | :---: | :---: |
| **Popularidad (Baseline)** | `0.0486 ±0.0128` | `0.0441` | `0.0269` |
| **Colaborativo (USER-KNN, k=30)** | `0.0045 ±0.0025` | `0.0047` | `0.0053` |
| **Matrix Factorization (SVD/SGD, d=40)** | `0.0681 ±0.0168` | `0.0606` | `0.0355` |
| **Graph RWR / Personalized PageRank (User-Movie [RQ1])** | `0.1082 ±0.0175` | `0.0859` | `0.0612` |
| **Graph RWR / Personalized PageRank (User-Movie-Genre [RQ2])** | `0.1082 ±0.0204` | `0.0859` | `0.0612` |
| **LightGCN (GNN 3-Layers, d=32)** | `0.1049 ±0.0196` | `0.0838` | `0.0588` |
