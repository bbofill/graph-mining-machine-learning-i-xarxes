# -*- coding: utf-8 -*-
import os
import csv
import zipfile
import urllib.request
from collections import defaultdict
from src.data.models import Movie, UserNode
from src.utils import Utils


class MovieLensDataset:
    """
    Gestiona la descarga, carga, limpieza y partición temporal (Train/Test)
    de los datasets MovieLens 1M y MovieLens Latest Small.
    """

    URLS = {
        "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    }

    def __init__(self, version="ml-1m", data_dir="data", min_user_ratings=10, min_item_ratings=5):
        self.version = version
        self.data_dir = data_dir
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings

        self.items = {}          # movie_id -> Movie
        self.users = {}          # user_id -> UserNode
        self.all_genres = set()  # Conjunto de todos los géneros disponibles
        self.raw_interactions = defaultdict(list)  # user_id -> [(item_id, rating, timestamp), ...]

        # Particiones después del split temporal:
        self.ratings = defaultdict(dict)       # Apunta a train_ratings para compatibilidad con Recommender
        self.train_ratings = defaultdict(dict) # user_id -> {item_id: rating}
        self.test_ratings = defaultdict(dict)  # user_id -> {item_id: rating}
        self.train_timestamps = defaultdict(dict)
        self.test_timestamps = defaultdict(dict)

        self._ensure_dataset_downloaded()
        self._load_data()
        self._filter_sparse_entities()
        self._validate_ratings()

    def _ensure_dataset_downloaded(self):
        """Descarga y descomprime automáticamente MovieLens si no existe en disco."""
        os.makedirs(self.data_dir, exist_ok=True)
        target_folder = os.path.join(self.data_dir, self.version)

        # Si el usuario ya tiene movies.csv y ratings.csv en la raíz (como en el código original), lo detectamos
        if self.version == "local-csv" and os.path.exists("ratings.csv") and os.path.exists("movies.csv"):
            Utils.log_message("INFO", "Detectados archivos locales ratings.csv y movies.csv en el directorio raíz.")
            return

        if not os.path.exists(target_folder):
            url = self.URLS.get(self.version)
            if not url:
                raise ValueError(f"Versión desconocida: {self.version}. Usa 'ml-1m' o 'ml-latest-small'.")
            zip_path = os.path.join(self.data_dir, f"{self.version}.zip")
            Utils.log_message("INFO", f"Descargando {self.version} desde {url}...")
            urllib.request.urlretrieve(url, zip_path)
            Utils.log_message("INFO", f"Descomprimiendo {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
            os.remove(zip_path)
            Utils.log_message("INFO", f"Dataset {self.version} listo en {target_folder}.")

    def _load_data(self):
        """Carga películas, usuarios y valoraciones con marcas de tiempo (timestamps)."""
        if self.version == "ml-1m":
            folder = os.path.join(self.data_dir, "ml-1m")
            movies_file = os.path.join(folder, "movies.dat")
            ratings_file = os.path.join(folder, "ratings.dat")
            users_file = os.path.join(folder, "users.dat")

            # 1. Cargar Películas (MovieID::Title::Genres)
            with open(movies_file, "r", encoding="latin-1") as f:
                for line in f:
                    parts = line.strip().split("::")
                    if len(parts) >= 3:
                        m_id, title, genres = str(parts[0]), parts[1], parts[2]
                        movie = Movie(m_id, title, genres)
                        self.items[m_id] = movie
                        self.all_genres.update(movie.genres)

            # 2. Cargar Usuarios (UserID::Gender::Age::Occupation::Zip-code)
            if os.path.exists(users_file):
                with open(users_file, "r", encoding="latin-1") as f:
                    for line in f:
                        parts = line.strip().split("::")
                        if len(parts) >= 4:
                            u_id = str(parts[0])
                            self.users[u_id] = UserNode(u_id, parts[1], parts[2], parts[3])

            # 3. Cargar Valoraciones (UserID::MovieID::Rating::Timestamp)
            total_rows = 0
            with open(ratings_file, "r", encoding="latin-1") as f:
                for line in f:
                    total_rows += 1
                    parts = line.strip().split("::")
                    if len(parts) >= 4:
                        u_id, m_id = str(parts[0]), str(parts[1])
                        rating, ts = float(parts[2]), int(parts[3])
                        if m_id in self.items:
                            self.raw_interactions[u_id].append((m_id, rating, ts))
                            if u_id not in self.users:
                                self.users[u_id] = UserNode(u_id)

            Utils.log_message("INFO", f"MovieLens 1M cargado: {total_rows} filas leídas.")

        else:
            # Formato CSV (ml-latest-small o archivos CSV locales)
            folder = "." if self.version == "local-csv" else os.path.join(self.data_dir, "ml-latest-small")
            movies_file = os.path.join(folder, "movies.csv")
            ratings_file = os.path.join(folder, "ratings.csv")

            with open(movies_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        m_id, title, genres = str(row[0]), row[1], row[2]
                        movie = Movie(m_id, title, genres)
                        self.items[m_id] = movie
                        self.all_genres.update(movie.genres)

            total_rows = 0
            with open(ratings_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    total_rows += 1
                    if len(row) >= 4:
                        u_id, m_id = str(row[0]), str(row[1])
                        rating, ts = float(row[2]), int(float(row[3]))
                        if m_id in self.items:
                            self.raw_interactions[u_id].append((m_id, rating, ts))
                            if u_id not in self.users:
                                self.users[u_id] = UserNode(u_id)

            Utils.log_message("INFO", f"MovieLens CSV ({self.version}) cargado: {total_rows} filas leídas.")

    def _filter_sparse_entities(self):
        """Filtra ítems y usuarios con menos de min_item_ratings y min_user_ratings."""
        item_counts = defaultdict(int)
        for u_id, inters in self.raw_interactions.items():
            for m_id, _, _ in inters:
                item_counts[m_id] += 1

        valid_items = {m_id for m_id, count in item_counts.items() if count >= self.min_item_ratings}

        filtered_interactions = defaultdict(list)
        for u_id, inters in self.raw_interactions.items():
            user_valid = [(m_id, r, ts) for (m_id, r, ts) in inters if m_id in valid_items]
            if len(user_valid) >= self.min_user_ratings:
                filtered_interactions[u_id] = user_valid

        self.raw_interactions = filtered_interactions
        active_items = {m_id for inters in self.raw_interactions.values() for m_id, _, _ in inters}
        self.items = {m_id: movie for m_id, movie in self.items.items() if m_id in active_items}
        self.users = {u_id: user for u_id, user in self.users.items() if u_id in self.raw_interactions}

    def _validate_ratings(self):
        all_r = [r for inters in self.raw_interactions.values() for _, r, _ in inters]
        if all_r:
            Utils.log_message(
                "INFO",
                f"Resumen Dataset ({self.version}): {len(self.users)} usuarios, "
                f"{len(self.items)} películas, {len(all_r)} valoraciones "
                f"(Rango: [{min(all_r)}, {max(all_r)}], Géneros: {len(self.all_genres)})."
            )

    def temporal_train_test_split(self, train_ratio=0.8):
        """
        Realiza la partición temporal estricta por usuario (Sección 5 del Plan).
        Para cada usuario, ordena sus valoraciones por timestamp ascendente:
        - El primer train_ratio (80%) va a Train.
        - El último (1 - train_ratio) (20%) va a Test.
        """
        self.train_ratings.clear()
        self.test_ratings.clear()
        self.train_timestamps.clear()
        self.test_timestamps.clear()

        total_train, total_test = 0, 0

        for u_id, inters in self.raw_interactions.items():
            # Ordenar estrictamente por timestamp (y por item_id en caso de empate)
            sorted_inters = sorted(inters, key=lambda x: (x[2], x[0]))
            split_idx = max(1, int(len(sorted_inters) * train_ratio))

            train_slice = sorted_inters[:split_idx]
            test_slice = sorted_inters[split_idx:]

            for m_id, rating, ts in train_slice:
                self.train_ratings[u_id][m_id] = rating
                self.train_timestamps[u_id][m_id] = ts
                total_train += 1

            for m_id, rating, ts in test_slice:
                # Solo evaluamos ítems que no estén ya en train del mismo usuario
                if m_id not in self.train_ratings[u_id]:
                    self.test_ratings[u_id][m_id] = rating
                    self.test_timestamps[u_id][m_id] = ts
                    total_test += 1

        # Por defecto, self.ratings apunta EXCLUSIVAMENTE a train_ratings para evitar data leakage
        self.ratings = self.train_ratings

        Utils.log_message(
            "INFO",
            f"Split Temporal ({int(train_ratio*100)}/{int((1-train_ratio)*100)}) completado: "
            f"Train = {total_train} valoraciones | Test = {total_test} valoraciones."
        )
        return self.train_ratings, self.test_ratings

    def get_item_features_dict(self):
        """Devuelve un diccionario {movie_id: 'Genre1|Genre2'} para ContentBasedRecommender."""
        return {m_id: movie.genres_text for m_id, movie in self.items.items()}
