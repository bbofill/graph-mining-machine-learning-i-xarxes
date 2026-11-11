# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from src.models.recommenders import Recommender
from src.graph.graph_builder import InteractionGraphBuilder
from src.utils import Utils


class PageRankGraphRecommender(Recommender):
    """
    Método Basado en Grafos: Personalized PageRank / Random Walk with Restart (RWR).
    Resuelve la ecuación estacionaria:
        pi_u = (1 - alpha) * P^T * pi_u + alpha * e_u
    mediante iteración de potencias sobre la matriz de transición del grafo.
    - Si include_genres=False -> Grafo Bipartito User-Movie (RQ1).
    - Si include_genres=True  -> Grafo Tripartito User-Movie-Genre (RQ2).
    """
    def __init__(self, dataset, graph_builder=None, include_genres=False, alpha=0.15, max_iter=25, tol=1e-6):
        suffix = "User-Movie-Genre [RQ2]" if include_genres else "User-Movie [RQ1]"
        super().__init__(dataset, name=f"Graph RWR / Personalized PageRank ({suffix})")
        self.include_genres = include_genres
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        self.builder = graph_builder if graph_builder else InteractionGraphBuilder(dataset)
        self.P_T = self.builder.build_transition_matrix(include_genres=include_genres).T.tocsr()
        self.num_total_nodes = self.P_T.shape[0]

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        if user_id not in self.builder.user_to_idx:
            return []

        u_idx = self.builder.user_to_idx[user_id]
        user_train = self.dataset.train_ratings[user_id]

        # Vector de reinicio personalizado (teleportación hacia el usuario activo y sus ítems favoritos)
        restart_vec = np.zeros(self.num_total_nodes, dtype=np.float32)
        restart_vec[u_idx] = 1.0

        pi = restart_vec.copy()
        for _ in range(self.max_iter):
            pi_next = (1.0 - self.alpha) * self.P_T.dot(pi) + self.alpha * restart_vec
            if np.linalg.norm(pi_next - pi, ord=1) < self.tol:
                pi = pi_next
                break
            pi = pi_next

        # Extraer las probabilidades estacionarias correspondientes a los nodos Película
        U = self.builder.num_users
        M = self.builder.num_items
        movie_probs = pi[U: U + M].copy()

        # Excluir películas ya valoradas en Train
        for m_id in user_train:
            if m_id in self.builder.item_to_idx:
                movie_probs[self.builder.item_to_idx[m_id]] = -np.inf

        top_indices = np.argpartition(movie_probs, -n)[-n:]
        top_indices = top_indices[np.argsort(movie_probs[top_indices])[::-1]]

        # Escalar puntuaciones para visualización interpretable junto con el ranking exacto
        max_prob = np.max(movie_probs[top_indices]) if len(top_indices) > 0 else 1.0
        max_prob = max_prob if max_prob > 0 else 1.0

        recs = []
        for idx in top_indices:
            if movie_probs[idx] > -np.inf:
                m_id = self.builder.item_ids[idx]
                scaled_score = 3.0 + 2.0 * (movie_probs[idx] / max_prob)
                recs.append((self.dataset.items[m_id], float(scaled_score)))
        return recs[:n]


class BipartiteResourceAllocationRecommender(Recommender):
    """
    Método Estructural de Grafos Bipartitos: Resource Allocation / Caminos de 3 Saltos
    normalizados por el grado de usuarios e ítems:
        S_RA = R * D_I^{-1} * R^T * D_U^{-1} * R
    Mitiga el sesgo de popularidad en el grafo bipartito User-Movie.
    """
    def __init__(self, dataset, graph_builder=None):
        super().__init__(dataset, name="Graph Resource Allocation (Bipartito 3-Hop)")
        self.builder = graph_builder if graph_builder else InteractionGraphBuilder(dataset)
        self._prepare_propagation_operators()

    def _prepare_propagation_operators(self):
        # Solo propagamos desde valoraciones positivas (>= 3.5) en el grafo de train
        R = self.builder.build_sparse_interaction_matrix(weighted=True, min_rating=3.5)
        deg_u = np.array(R.sum(axis=1)).flatten()
        deg_i = np.array(R.sum(axis=0)).flatten()

        inv_deg_u = np.divide(1.0, deg_u, out=np.zeros_like(deg_u), where=deg_u > 0)
        inv_deg_i = np.divide(1.0, deg_i, out=np.zeros_like(deg_i), where=deg_i > 0)

        self.R_norm_u = sp.diags(inv_deg_u).dot(R).tocsr()
        self.R_norm_i = R.dot(sp.diags(inv_deg_i)).tocsr()

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        if user_id not in self.builder.user_to_idx:
            return []

        u_idx = self.builder.user_to_idx[user_id]
        user_train = self.dataset.train_ratings[user_id]

        # Salto 1: Usuario -> Películas vistas
        u_items = self.R_norm_i[u_idx]
        # Salto 2: Películas vistas -> Otros Usuarios con gustos estructuralmente afines
        u_neighbors = u_items.dot(self.R_norm_u.T)
        # Salto 3: Vecinos en el grafo -> Películas candidatas
        scores = u_neighbors.dot(self.R_norm_i).toarray().flatten()

        for m_id in user_train:
            if m_id in self.builder.item_to_idx:
                scores[self.builder.item_to_idx[m_id]] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        max_s = np.max(scores[top_indices]) if len(top_indices) > 0 and np.max(scores[top_indices]) > 0 else 1.0
        return [
            (self.dataset.items[self.builder.item_ids[idx]], float(3.0 + 2.0 * (scores[idx] / max_s)))
            for idx in top_indices if scores[idx] > -np.inf
        ]


class LightGCNRecommender(Recommender):
    """
    Graph Neural Network (Sección 4 y Fase 6 del Plan): LightGCN.
    Aprende embeddings E^(0) de usuarios y películas propagándolos linealmente
    a través de K capas sobre la matriz de adyacencia normalizada del grafo:
        E^(k+1) = (D^{-1/2} A D^{-1/2}) E^(k)
        E_final = (1 / (K + 1)) * sum_{k=0}^K E^(k)
    Optimizado mediante Bayesian Personalized Ranking (BPR Loss).
    """
    def __init__(self, dataset, graph_builder=None, emb_dim=32, n_layers=3, lr=0.02, reg=1e-4, epochs=25, batch_size=2048):
        super().__init__(dataset, name=f"LightGCN (GNN {n_layers}-Layers, d={emb_dim})")
        self.builder = graph_builder if graph_builder else InteractionGraphBuilder(dataset)
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size

        self.A_hat, self.R_pos = self.builder.build_normalized_laplacian_adjacency(min_rating=3.5)
        self._train_lightgcn()

    def _propagate(self, E0):
        """Propaga los embeddings a través de las K capas del grafo y promedia."""
        all_embeddings = [E0]
        E_k = E0
        for _ in range(self.n_layers):
            E_k = self.A_hat.dot(E_k)
            all_embeddings.append(E_k)
        E_final = np.mean(all_embeddings, axis=0)
        return E_final, all_embeddings

    def _train_lightgcn(self):
        U, M = self.builder.num_users, self.builder.num_items
        # Inicialización Xavier/Normal de embeddings de capa 0
        self.E0 = np.random.normal(0.0, 0.1, size=(U + M, self.emb_dim)).astype(np.float32)

        # Extraer pares positivos (u, i_pos) de train
        pos_u, pos_i = self.R_pos.nonzero()
        num_pos = len(pos_u)
        if num_pos == 0:
            self.user_final_emb = self.E0[:U]
            self.item_final_emb = self.E0[U:]
            return

        user_pos_sets = [set() for _ in range(U)]
        for u, i in zip(pos_u, pos_i):
            user_pos_sets[u].add(i)

        # Optimizador Adam sobre los embeddings E0 con propagación en el grafo
        m_adam = np.zeros_like(self.E0)
        v_adam = np.zeros_like(self.E0)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        for _ in range(self.epochs):
            perm = np.random.permutation(num_pos)
            for start in range(0, num_pos, self.batch_size):
                step += 1
                batch_idx = perm[start: start + self.batch_size]
                u_batch = pos_u[batch_idx]
                i_pos_batch = pos_i[batch_idx]

                # Muestreo negativo uniforme (ítems no conectados al usuario en train)
                i_neg_batch = np.random.randint(0, M, size=len(u_batch))
                for idx_b, u_val in enumerate(u_batch):
                    while i_neg_batch[idx_b] in user_pos_sets[u_val]:
                        i_neg_batch[idx_b] = np.random.randint(0, M)

                # Forward pass: Propagación multicapa en el grafo
                E_final, _ = self._propagate(self.E0)
                u_emb = E_final[u_batch]
                pos_emb = E_final[U + i_pos_batch]
                neg_emb = E_final[U + i_neg_batch]

                # Puntuación diferencial BPR: x_uij = <e_u, e_i> - <e_u, e_j>
                x_uij = np.sum(u_emb * (pos_emb - neg_emb), axis=1)
                sigmoid_neg = 1.0 / (1.0 + np.exp(np.clip(x_uij, -20, 20)))  # d(-ln sigma(x))/dx = -(1 - sigma(x))

                # Gradientes respecto a E_final
                grad_final = np.zeros_like(E_final)
                coeff = -sigmoid_neg[:, np.newaxis] / len(u_batch)

                np.add.at(grad_final, u_batch, coeff * (pos_emb - neg_emb))
                np.add.at(grad_final, U + i_pos_batch, coeff * u_emb)
                np.add.at(grad_final, U + i_neg_batch, -coeff * u_emb)

                # Backpropagation a través de las capas del grafo (como A_hat es simétrica, el adjunto es A_hat)
                grad_E0 = grad_final.copy()
                g_k = grad_final
                for _ in range(self.n_layers):
                    g_k = self.A_hat.dot(g_k)
                    grad_E0 += g_k
                grad_E0 /= (self.n_layers + 1)

                # Regularización L2 sobre E0
                grad_E0 += self.reg * self.E0

                # Actualización Adam
                m_adam = beta1 * m_adam + (1.0 - beta1) * grad_E0
                v_adam = beta2 * v_adam + (1.0 - beta2) * (grad_E0 ** 2)
                m_hat = m_adam / (1.0 - beta1 ** step)
                v_hat = v_adam / (1.0 - beta2 ** step)
                self.E0 -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        E_final, _ = self._propagate(self.E0)
        self.user_final_emb = E_final[:U]
        self.item_final_emb = E_final[U:]
        Utils.log_message("INFO", f"{self.name} entrenado mediante BPR Loss ({self.epochs} épocas).")

    def recommend(self, user_id, n=10):
        user_id = str(user_id)
        if user_id not in self.builder.user_to_idx:
            return []

        u_idx = self.builder.user_to_idx[user_id]
        user_train = self.dataset.train_ratings[user_id]

        u_vec = self.user_final_emb[u_idx]
        scores = np.dot(self.item_final_emb, u_vec)

        for m_id in user_train:
            if m_id in self.builder.item_to_idx:
                scores[self.builder.item_to_idx[m_id]] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (self.dataset.items[self.builder.item_ids[idx]], float(scores[idx]))
            for idx in top_indices if scores[idx] > -np.inf
        ]
