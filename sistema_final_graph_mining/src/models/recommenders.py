# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import defaultdict
import statistics
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from src.utils import Utils


class Recommender(ABC):
    """Clase base abstracta para todos los sistemas de recomendación."""
    def __init__(self, dataset, min_vots=5, name="BaseRecommender"):
        self.dataset = dataset
        self.min_vots = min_vots
        self.name = name
        self.item_stats = {}
        self.avg_global = 0.0
        self._calculate_item_stats()

    def _calculate_item_stats(self):
        item_ratings = defaultdict(list)
        all_ratings = []
        for user_ratings in self.dataset.train_ratings.values():
            for item_id, rating in user_ratings.items():
                if rating > 0:
                    item_ratings[item_id].append(rating)
                    all_ratings.append(rating)

        self.avg_global = statistics.mean(all_ratings) if all_ratings else 3.5
        self.item_stats = {}
        for item_id, votes in item_ratings.items():
            if len(votes) >= self.min_vots and item_id in self.dataset.items:
                self.item_stats[item_id] = {
                    "avg_item": statistics.mean(votes),
                    "num_vots": len(votes)
                }

    def get_unrated_eligible_items(self, user_id):
        """Devuelve los IDs de películas que el usuario NO ha valorado en train."""
        user_train = self.dataset.train_ratings.get(str(user_id), {})
        return [
            m_id for m_id in self.dataset.items
            if m_id not in user_train
        ]

    @abstractmethod
    def recommend(self, user_id, n=10):
        """Devuelve una lista ordenada [(Movie, score), ...] de longitud máxima n."""
        pass


class PopularityRecommender(Recommender):
    """
    1. Baseline de Popularidad (evolución de SimpleRecommender).
    Calcula la puntuación ponderada de IMDB (Weighted Rating) usando solo datos de Train.
    """
    def __init__(self, dataset, min_vots=20):
        super().__init__(dataset, min_vots=min_vots, name="Popularidad (Baseline)")
        self._precompute_popular_ranking()

    def _calculate_score(self, avg_item, num_vots):
        denom = num_vots + self.min_vots
        if denom == 0:
            return 0.0
        return (num_vots / denom) * avg_item + (self.min_vots / denom) * self.avg_global

    def _precompute_popular_ranking(self):
        scored = []
        for item_id, stats in self.item_stats.items():
            score = self._calculate_score(stats["avg_item"], stats["num_vots"])
            scored.append((item_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        self.global_ranking = scored

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        user_train = self.dataset.train_ratings.get(user_id, {})
        recs = []
        for item_id, score in self.global_ranking:
            if item_id not in user_train:
                recs.append((self.dataset.items[item_id], float(score)))
                if len(recs) >= n:
                    break
        return recs


class CollaborativeRecommender(Recommender):
    """
    2. Filtrado Colaborativo Basado en Vecinos (User-KNN o Item-KNN).
    Evolución optimizada con álgebra matricial dispersa de tu CollaborativeRecommender original.
    """
    def __init__(self, dataset, mode="user", k=30, min_vots=5):
        model_label = f"Colaborativo ({mode.upper()}-KNN, k={k})"
        super().__init__(dataset, min_vots=min_vots, name=model_label)
        self.mode = mode.lower()
        self.k = k
        self._build_similarity_structures()

    def _build_similarity_structures(self):
        self.user_ids = sorted(list(self.dataset.train_ratings.keys()))
        self.item_ids = sorted(list(self.dataset.items.keys()))
        self.u_map = {u: i for i, u in enumerate(self.user_ids)}
        self.i_map = {m: i for i, m in enumerate(self.item_ids)}

        rows, cols, vals = [], [], []
        self.user_means = np.full(len(self.user_ids), self.avg_global, dtype=np.float32)

        for u_id, ratings in self.dataset.train_ratings.items():
            u_idx = self.u_map[u_id]
            if ratings:
                u_mean = np.mean(list(ratings.values()))
                self.user_means[u_idx] = u_mean
                for m_id, r in ratings.items():
                    if m_id in self.i_map:
                        rows.append(u_idx)
                        cols.append(self.i_map[m_id])
                        vals.append(r)

        self.R = sp.csr_matrix((vals, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)), dtype=np.float32)

        # Matriz centrada por usuario (Mean-Centered Ratings) y normalizada por norma L2
        R_centered = self.R.copy().tolil()
        for u_idx in range(len(self.user_ids)):
            nz = R_centered.rows[u_idx]
            if nz:
                mean_val = self.user_means[u_idx]
                R_centered.data[u_idx] = [v - mean_val for v in R_centered.data[u_idx]]
        self.R_centered = R_centered.tocsr()

        if self.mode == "user":
            norms = sp.linalg.norm(self.R, axis=1)
            norms[norms == 0] = 1.0
            self.R_norm_user = sp.diags(1.0 / norms).dot(self.R).tocsr()
        else:
            norms = sp.linalg.norm(self.R, axis=0)
            norms[norms == 0] = 1.0
            self.R_norm_item = self.R.dot(sp.diags(1.0 / norms)).tocsc()

        Utils.log_message("INFO", f"{self.name} inicializado sobre matriz {self.R.shape}.")

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        if user_id not in self.u_map:
            return []

        u_idx = self.u_map[user_id]
        u_mean = float(self.user_means[u_idx])
        user_train = self.dataset.train_ratings[user_id]

        if self.mode == "user":
            # Similitud coseno del usuario activo con todos los demás usuarios
            u_vec = self.R_norm_user[u_idx]
            sims = self.R_norm_user.dot(u_vec.T).toarray().flatten()
            sims[u_idx] = 0.0  # Excluir auto-similitud

            # Seleccionar los top-K vecinos más similares
            top_k_users = np.argpartition(sims, -self.k)[-self.k:]
            top_sims = sims[top_k_users]
            valid_mask = top_sims > 0
            top_k_users = top_k_users[valid_mask]
            top_sims = top_sims[valid_mask]

            if len(top_k_users) == 0:
                return []

            # Predicción ponderada sobre las desviaciones respecto a la media de cada vecino
            neighbor_centered = self.R_centered[top_k_users]
            numerator = top_sims.dot(neighbor_centered.toarray())

            # Denominador: suma de similitudes de vecinos que sí valoraron cada ítem
            neighbor_binary = (self.R[top_k_users] > 0).astype(np.float32)
            denominator = top_sims.dot(neighbor_binary.toarray())

            scores = np.full(len(self.item_ids), u_mean, dtype=np.float32)
            nonzero = denominator > 1e-6
            scores[nonzero] = u_mean + (numerator[nonzero] / denominator[nonzero])
            # Penalizar ítems que ningún vecino entre los K ha visto
            scores[~nonzero] = -1.0

        else:
            # Modo Item-KNN
            rated_indices = [self.i_map[m] for m in user_train if m in self.i_map]
            if not rated_indices:
                return []
            user_ratings_vec = np.array([user_train[self.item_ids[idx]] - u_mean for idx in rated_indices], dtype=np.float32)
            rated_item_vecs = self.R_norm_item[:, rated_indices]
            # Matriz de similitud ítem-ítem respecto a los ítems valorados por el usuario
            item_sims = self.R_norm_item.T.dot(rated_item_vecs).toarray()  # (num_items x len(rated))

            numerator = item_sims.dot(user_ratings_vec)
            denominator = np.sum(np.abs(item_sims), axis=1)
            scores = np.full(len(self.item_ids), -1.0, dtype=np.float32)
            nonzero = denominator > 1e-6
            scores[nonzero] = u_mean + (numerator[nonzero] / denominator[nonzero])

        # Filtrar películas ya vistas en Train y ordenar Top-N
        for m_id in user_train:
            if m_id in self.i_map:
                scores[self.i_map[m_id]] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        recs = []
        for idx in top_indices:
            if scores[idx] > -np.inf:
                m_id = self.item_ids[idx]
                clipped_score = float(np.clip(scores[idx], 0.5, 5.0))
                recs.append((self.dataset.items[m_id], clipped_score))
        return recs[:n]


class MatrixFactorizationRecommender(Recommender):
    """
    3. Matrix Factorization (Sección 4 del Plan).
    Aprende factores latentes de usuarios (P) y películas (Q) con sesgos:
    r_hat(u, i) = mu + b_u + b_i + P_u · Q_i^T
    Combina inicialización espectral SVD con refinamiento por SGD regularizado.
    """
    def __init__(self, dataset, n_factors=40, lr=0.008, reg=0.05, n_epochs=15):
        super().__init__(dataset, name=f"Matrix Factorization (SVD/SGD, d={n_factors})")
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self._fit()

    def _fit(self):
        self.user_ids = sorted(list(self.dataset.train_ratings.keys()))
        self.item_ids = sorted(list(self.dataset.items.keys()))
        self.u_map = {u: i for i, u in enumerate(self.user_ids)}
        self.i_map = {m: i for i, m in enumerate(self.item_ids)}

        n_u, n_i = len(self.user_ids), len(self.item_ids)
        self.mu = float(self.avg_global)
        self.b_u = np.zeros(n_u, dtype=np.float32)
        self.b_i = np.zeros(n_i, dtype=np.float32)

        # Recopilar muestras de entrenamiento (u_idx, i_idx, rating)
        samples_u, samples_i, samples_r = [], [], []
        for u_id, ratings in self.dataset.train_ratings.items():
            u_idx = self.u_map[u_id]
            for m_id, r in ratings.items():
                if m_id in self.i_map:
                    samples_u.append(u_idx)
                    samples_i.append(self.i_map[m_id])
                    samples_r.append(float(r))

        samples_u = np.array(samples_u, dtype=np.int32)
        samples_i = np.array(samples_i, dtype=np.int32)
        samples_r = np.array(samples_r, dtype=np.float32)

        # Inicialización mediante SVD truncado sobre la matriz centrada
        R_centered = sp.csr_matrix(
            (samples_r - self.mu, (samples_u, samples_i)),
            shape=(n_u, n_i),
            dtype=np.float32
        )
        k = min(self.n_factors, min(n_u, n_i) - 2)
        u_svd, s_svd, vt_svd = svds(R_centered, k=k)
        sqrt_s = np.sqrt(s_svd)
        self.P = (u_svd * sqrt_s).astype(np.float32)
        self.Q = (vt_svd.T * sqrt_s).astype(np.float32)

        # Refinamiento con SGD vectorizado por épocas
        num_samples = len(samples_r)
        indices = np.arange(num_samples)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for idx in indices:
                u = samples_u[idx]
                i = samples_i[idx]
                r = samples_r[idx]

                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred

                # Actualizar sesgos y factores latentes
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])

        Utils.log_message("INFO", f"{self.name} entrenado con {num_samples} interacciones ({self.n_epochs} épocas).")

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        if user_id not in self.u_map:
            return []

        u_idx = self.u_map[user_id]
        user_train = self.dataset.train_ratings[user_id]

        # Predicción rápida para todos los ítems: mu + b_u + b_i + P[u] @ Q.T
        scores = self.mu + self.b_u[u_idx] + self.b_i + np.dot(self.Q, self.P[u_idx])

        for m_id in user_train:
            if m_id in self.i_map:
                scores[self.i_map[m_id]] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (self.dataset.items[self.item_ids[idx]], float(np.clip(scores[idx], 0.5, 5.0)))
            for idx in top_indices if scores[idx] > -np.inf
        ]


class ContentBasedRecommender(Recommender):
    """
    4. Recomendador Basado en Contenido (TF-IDF sobre géneros).
    Conservado y optimizado a partir de tu implementación original.
    """
    def __init__(self, dataset, item_features=None, min_vots=3):
        super().__init__(dataset, min_vots=min_vots, name="Contenido (TF-IDF Géneros)")
        self.item_features = item_features if item_features else dataset.get_item_features_dict()
        self._compute_tfidf_matrix()

    def _compute_tfidf_matrix(self):
        self.item_ids = sorted(list(self.item_features.keys()))
        self.i_map = {m: i for i, m in enumerate(self.item_ids)}
        vocab_set = set()
        for feat in self.item_features.values():
            tokens = feat.split("|") if "|" in feat else feat.split()
            vocab_set.update([t for t in tokens if t])

        self.vocab = sorted(list(vocab_set))
        v_map = {w: i for i, w in enumerate(self.vocab)}

        self.tfidf_matrix = np.zeros((len(self.item_ids), len(self.vocab)), dtype=np.float32)
        doc_freq = np.zeros(len(self.vocab), dtype=np.float32)

        for i, m_id in enumerate(self.item_ids):
            feat = self.item_features[m_id]
            tokens = [t for t in (feat.split("|") if "|" in feat else feat.split()) if t in v_map]
            if not tokens:
                continue
            counts = defaultdict(int)
            for t in tokens:
                counts[v_map[t]] += 1
            for w_idx, c in counts.items():
                self.tfidf_matrix[i, w_idx] = c / len(tokens)
                doc_freq[w_idx] += 1.0

        idf = np.log(len(self.item_ids) / (doc_freq + 1.0))
        self.tfidf_matrix *= idf

        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_normalized = self.tfidf_matrix / norms

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        user_train = self.dataset.train_ratings.get(user_id, {})
        if not user_train:
            return []

        idxs, weights = [], []
        for m_id, r in user_train.items():
            if m_id in self.i_map and r > 0:
                idxs.append(self.i_map[m_id])
                weights.append(r)

        if not idxs:
            return []

        weights = np.array(weights, dtype=np.float32)
        profile = np.sum(self.tfidf_normalized[idxs] * weights[:, np.newaxis], axis=0)
        norm_p = np.linalg.norm(profile)
        if norm_p == 0:
            return []
        profile /= norm_p

        sims = np.dot(self.tfidf_normalized, profile)
        for m_id in user_train:
            if m_id in self.i_map:
                sims[self.i_map[m_id]] = -np.inf

        top_indices = np.argpartition(sims, -n)[-n:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [
            (self.dataset.items[self.item_ids[idx]], float(np.clip(sims[idx] * 5.0, 0.5, 5.0)))
            for idx in top_indices if sims[idx] > -np.inf
        ]
