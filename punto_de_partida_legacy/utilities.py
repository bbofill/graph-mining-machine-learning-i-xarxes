# -*- coding: utf-8 -*-
import time as t

class Utils:
    """Clase auxiliar para funcionalidades de logging y persistencia."""
    
    @staticmethod
    def log_message(level, message):
        """Registra mensajes en consola y, si es WARNING o ERROR, en un archivo de log."""
        timestamp = t.time()
        print(f"{timestamp} - {level} - {message}")
        if level in ["WARNING", "ERROR"]:
            with open(f"log_{timestamp}.txt", "a") as log_file:
                log_file.write(f"{timestamp} - {level} - {message}\n")