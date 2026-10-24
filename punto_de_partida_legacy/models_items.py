# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class Item(ABC):
    """Clase base abstracta para ítems como películas o libros."""
    def __init__(self, item_id):
        self._id = item_id

    @property
    def id(self):
        return self._id

    @abstractmethod
    def get_info(self):
        pass


class Movie(Item):
    """Clase para representar una película identificada por su ID."""
    def __init__(self, movie_id, name):
        super().__init__(movie_id)
        self._name = name

    def get_info(self):
        return f"Película: {self._name}"


class Book(Item):
    """Clase para representar un libro identificado por su ISBN."""
    def __init__(self, isbn, name):
        super().__init__(isbn)
        self._name = name

    def get_info(self):
        return f"Libro: {self._name}"