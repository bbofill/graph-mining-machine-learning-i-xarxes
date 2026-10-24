# -*- coding: utf-8 -*-
import numpy as np

class Evaluator:
    """Clase para evaluar sistemas de recomendación usando MAE y RMSE."""
    def __init__(self, recommender, test_ratings):
        self.recommender = recommender
        self.test_ratings = test_ratings

    def evaluate(self, user_id, n=5):
        predictions = []
        actual = []

        if user_id not in self.test_ratings:
            return [], [], 0.0, 0.0

        test_items = list(self.test_ratings[user_id].keys())
        recommendations = self.recommender.recommend(user_id, len(self.recommender.dataset.items))

        for item, score in recommendations:
            item_id = item.id
            if item_id in test_items:
                predictions.append(score)
                actual.append(self.test_ratings[user_id][item_id])
                if len(predictions) >= n:
                    break

        if not predictions:
            return [], [], 0.0, 0.0

        predictions = np.array(predictions)
        actual = np.array(actual)
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        return predictions.tolist(), actual.tolist(), mae, rmse