# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import re


class Item(ABC):
    """Clase base abstracta para ítems recomendables (películas o libros)."""
    def __init__(self, item_id):
        self._id = str(item_id)

    @property
    def id(self):
        return self._id

    @abstractmethod
    def get_info(self):
        pass


class Movie(Item):
    """Clase para representar una película con su ID, título, año y lista de géneros."""
    def __init__(self, movie_id, name, genres_str=""):
        super().__init__(movie_id)
        self._name = name
        self._genres = [g.strip() for g in genres_str.split("|") if g.strip() and g != "(no genres listed)"]
        self._year = self._extract_year(name)

    @staticmethod
    def _extract_year(title):
        match = re.search(r"\((\d{4})\)", title)
        return int(match.group(1)) if match else None

    @property
    def name(self):
        return self._name

    @property
    def genres(self):
        return self._genres

    @property
    def year(self):
        return self._year

    @property
    def genres_text(self):
        return "|".join(self._genres)

    def get_info(self):
        genres_fmt = ", ".join(self._genres) if self._genres else "Sin género"
        return f"Película [{self._id}]: {self._name} | Géneros: {genres_fmt}"


class Book(Item):
    """Clase para representar un libro identificado por su ISBN (compatibilidad legado)."""
    def __init__(self, isbn, name, author=""):
        super().__init__(isbn)
        self._name = name
        self._author = author

    @property
    def name(self):
        return self._name

    def get_info(self):
        return f"Libro [{self._id}]: {self._name}"


class UserNode:
    """Representa un nodo Usuario en el grafo de interacciones."""
    def __init__(self, user_id, gender=None, age=None, occupation=None):
        self.id = str(user_id)
        self.gender = gender
        self.age = age
        self.occupation = occupation

    def get_info(self):
        return f"Usuario [{self.id}] (Género: {self.gender}, Edad: {self.age})"
