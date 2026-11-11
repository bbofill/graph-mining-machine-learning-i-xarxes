# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.sparse as sp
from src.utils import Utils


class InteractionGraphBuilder:
    """
    Construye representaciones en grafo a partir de las interacciones de entrenamiento:
    1. Grafo Bipartito User-Movie (RQ1).
    2. Grafo Tripartito/Heterogéneo User-Movie-Genre (RQ2).
    """

    def __init__(self, dataset):
        self.dataset = dataset

        # Índices enteros para operaciones matriciales rápidas
        self.user_ids = sorted(list(dataset.train_ratings.keys()))
        self.item_ids = sorted(list(dataset.items.keys()))
        self.genre_names = sorted(list(dataset.all_genres))

        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {m: i for i, m in enumerate(self.item_ids)}
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_names)}

        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.num_genres = len(self.genre_names)

    def build_networkx_graph(self, include_genres=False, min_rating_edge=0.0):
        """
        Construye un grafo de NetworkX con nodos prefijados:
        - 'U_<id>' para usuarios
        - 'M_<id>' para películas
        - 'G_<nombre>' para géneros (si include_genres=True)
        """
        G = nx.Graph()

        # Añadir nodos de usuario
        for u_id in self.user_ids:
            G.add_node(f"U_{u_id}", bipartite=0, node_type="user", raw_id=u_id)

        # Añadir nodos de película
        for m_id in self.item_ids:
            movie = self.dataset.items[m_id]
            G.add_node(
                f"M_{m_id}",
                bipartite=1,
                node_type="movie",
                raw_id=m_id,
                title=movie.name,
                genres=movie.genres_text
            )

        # Añadir aristas User --ha valorado--> Movie (solo de TRAIN)
        edge_count = 0
        for u_id, rated_movies in self.dataset.train_ratings.items():
            for m_id, rating in rated_movies.items():
                if rating >= min_rating_edge and m_id in self.item_to_idx:
                    G.add_edge(
                        f"U_{u_id}",
                        f"M_{m_id}",
                        weight=float(rating),
                        edge_type="rated"
                    )
                    edge_count += 1

        # Ampliación RQ2: Añadir nodos de Género y aristas Movie --pertenece_a--> Genre
        genre_edges = 0
        if include_genres:
            for g_name in self.genre_names:
                G.add_node(f"G_{g_name}", bipartite=2, node_type="genre", raw_id=g_name, title=g_name)

            for m_id in self.item_ids:
                movie = self.dataset.items[m_id]
                for g_name in movie.genres:
                    if g_name in self.genre_to_idx:
                        # MEJORA RQ2: Peso estabilizado en 1.0 para no sobreponderar los géneros
                        G.add_edge(
                            f"M_{m_id}",
                            f"G_{g_name}",
                            weight=1.0,
                            edge_type="has_genre"
                        )
                        genre_edges += 1

        graph_type = "User-Movie-Genre" if include_genres else "User-Movie"
        Utils.log_message(
            "INFO",
            f"Grafo NetworkX [{graph_type}] construido: {G.number_of_nodes()} nodos, "
            f"{edge_count} aristas User-Movie, {genre_edges} aristas Movie-Genre."
        )
        return G

    def build_sparse_interaction_matrix(self, weighted=True, min_rating=0.0):
        """
        Construye la matriz dispersa R de dimensiones (num_users x num_items)
        usando exclusivamente las valoraciones de entrenamiento.
        """
        rows, cols, vals = [], [], []
        for u_id, rated_movies in self.dataset.train_ratings.items():
            if u_id not in self.user_to_idx:
                continue
            u_idx = self.user_to_idx[u_id]
            for m_id, rating in rated_movies.items():
                if m_id in self.item_to_idx and rating >= min_rating:
                    m_idx = self.item_to_idx[m_id]
                    rows.append(u_idx)
                    cols.append(m_idx)
                    vals.append(float(rating) if weighted else 1.0)

        R = sp.csr_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return R

    def build_transition_matrix(self, include_genres=False, genre_weight=0.3):
        """
        Construye la matriz de transición estocástica por filas P = D^{-1} A
        para Random Walk with Restart / Personalized PageRank matricial rápido.
        Nodos ordenados como: [Usuarios (0..U-1), Películas (U..U+M-1), Géneros (U+M..U+M+G-1)]
        
        MEJORA RQ2: genre_weight=0.3 concentra la probabilidad en la relación User-Movie
        evitando que el paseo aleatorio se diluya excesivamente por nodos de género muy populares.
        """
        R = self.build_sparse_interaction_matrix(weighted=True)
        U, M, G_cnt = self.num_users, self.num_items, self.num_genres

        if not include_genres:
            # Matriz de adyacencia bipartita (U+M) x (U+M)
            zero_u = sp.csr_matrix((U, U), dtype=np.float32)
            zero_m = sp.csr_matrix((M, M), dtype=np.float32)
            A = sp.bmat([[zero_u, R], [R.T, zero_m]], format="csr")
        else:
            # Matriz Movie-Genre (M x G)
            g_rows, g_cols, g_vals = [], [], []
            for m_id, m_idx in self.item_to_idx.items():
                for g_name in self.dataset.items[m_id].genres:
                    if g_name in self.genre_to_idx:
                        g_rows.append(m_idx)
                        g_cols.append(self.genre_to_idx[g_name])
                        g_vals.append(float(genre_weight))
            MG = sp.csr_matrix((g_vals, (g_rows, g_cols)), shape=(M, G_cnt), dtype=np.float32)

            zero_u = sp.csr_matrix((U, U), dtype=np.float32)
            zero_ug = sp.csr_matrix((U, G_cnt), dtype=np.float32)
            zero_m = sp.csr_matrix((M, M), dtype=np.float32)
            zero_g = sp.csr_matrix((G_cnt, G_cnt), dtype=np.float32)

            A = sp.bmat([
                [zero_u, R, zero_ug],
                [R.T, zero_m, MG],
                [zero_ug.T, MG.T, zero_g]
            ], format="csr")

        # Normalización por filas: P = D^{-1} A
        row_sums = np.array(A.sum(axis=1)).flatten()
        inv_deg = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        D_inv = sp.diags(inv_deg)
        P = D_inv.dot(A).tocsr()
        return P

    def build_normalized_laplacian_adjacency(self, min_rating=3.5):
        """
        Construye la matriz de adyacencia simétricamente normalizada:
        A_hat = D^{-1/2} A D^{-1/2}
        utilizada por LightGCN para propagar embeddings entre usuarios y películas.
        """
        # En LightGCN las aristas representan interacciones positivas observadas en train
        R = self.build_sparse_interaction_matrix(weighted=False, min_rating=min_rating)
        U, M = self.num_users, self.num_items
        zero_u = sp.csr_matrix((U, U), dtype=np.float32)
        zero_m = sp.csr_matrix((M, M), dtype=np.float32)
        A = sp.bmat([[zero_u, R], [R.T, zero_m]], format="csr")

        degrees = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        d_inv_sqrt[degrees == 0] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocsr()
        return A_hat, R
