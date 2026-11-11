# -*- coding: utf-8 -*-
import os
import json
import random
import datetime
import numpy as np


class Utils:
    """Clase auxiliar para logging, reproducibilidad y persistencia de experimentos."""

    LOG_DIR = "results/logs"

    @staticmethod
    def setup_directories():
        """Crea la estructura de directorios para datos y resultados."""
        for folder in ["data", "results", "results/logs", "results/plots", "results/graphs"]:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """Fija las semillas aleatorias para garantizar la reproducibilidad del experimento."""
        random.seed(seed)
        np.random.seed(seed)
        Utils.log_message("INFO", f"Semilla aleatoria fijada en: {seed}")

    @staticmethod
    def log_message(level, message):
        """Registra mensajes en consola y guarda WARNING/ERROR/INFO en archivo de log."""
        os.makedirs(Utils.LOG_DIR, exist_ok=True)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{now}] [{level}] {message}"
        print(formatted_msg)

        log_filename = os.path.join(Utils.LOG_DIR, "experiment_log.txt")
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(formatted_msg + "\n")

    @staticmethod
    def save_experiment_config(config_dict, filename="results/experiment_config.json"):
        """Guarda la configuración e hiperparámetros del protocolo experimental."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        Utils.log_message("INFO", f"Configuración guardada en {filename}")
