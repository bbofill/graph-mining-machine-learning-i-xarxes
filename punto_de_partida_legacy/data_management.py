# -*- coding: utf-8 -*-
import csv
from io import StringIO
from collections import defaultdict
from models import Movie, Book
from utils import Utils

class Dataset:
    """Clase para gestionar un conjunto de datos de valoraciones."""
    
    def __init__(self, name, ratings_csv, user_col, item_col, rating_col, has_timestamp=False, item_names=None):
        self.name = name
        self.items = {}
        self.ratings = self._load_ratings(ratings_csv, user_col, item_col, rating_col, has_timestamp)
        self._populate_items(item_names)
        self._validate_ratings()

    def _populate_items(self, item_names):
        for user_ratings in self.ratings.values():
            for item_id in user_ratings:
                if item_id not in self.items:
                    name = item_names.get(item_id, f"Desconocido_{item_id}") if item_names else f"Desconocido_{item_id}"
                    if self.name == "Películas":
                        self.items[item_id] = Movie(item_id, name)
                    elif self.name == "Libros":
                        self.items[item_id] = Book(item_id, name)

    def _load_ratings(self, ratings_csv, user_col, item_col, rating_col, has_timestamp):
        ratings = defaultdict(dict)
        csv_file = StringIO(ratings_csv)
        csvreader = csv.reader(csv_file)
        header = next(csvreader, None)
        if not header:
            Utils.log_message("ERROR", f"El archivo CSV para {self.name} está vacío.")
            return ratings

        total_rows = 0
        skipped_rows = 0
        for row in csvreader:
            total_rows += 1
            if has_timestamp and len(row) < 4:
                skipped_rows += 1
                continue
            if not has_timestamp and len(row) < 3:
                skipped_rows += 1
                continue
            try:
                user_id = str(row[0])
                item_id = str(row[1])
                rating = float(row[2])
                ratings[user_id][item_id] = rating
            except (ValueError, IndexError):
                skipped_rows += 1
                continue

        Utils.log_message("INFO", f"Dataset {self.name}: {total_rows} filas procesadas, {skipped_rows} filas omitidas, {len(ratings)} usuarios cargados.")
        return ratings

    def _validate_ratings(self):
        all_ratings = [rating for user_ratings in self.ratings.values() for rating in user_ratings.values()]
        if all_ratings:
            min_rating, max_rating = min(all_ratings), max(all_ratings)
            Utils.log_message("INFO", f"Validación de ratings ({self.name}): Mínimo: {min_rating}, Máximo: {max_rating}")