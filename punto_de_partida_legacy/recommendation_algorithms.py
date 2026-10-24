# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import statistics
import math
import numpy as np
from collections import defaultdict
from utils import Utils

class Recommender(ABC):
    """Clase base abstracta para sistemas de recomendación."""
    def __init__(self, dataset, min_vots=3):
        self.dataset = dataset
        self.min_vots = min_vots
        self.item_stats = None
        self.avg_global = None

    def _calculate_item_stats(self):
        item_ratings = defaultdict(list)
        all_ratings = []
        for user_ratings in self.dataset.ratings.values():
            for item_id, rating in user_ratings.items():
                if rating > 0:
                    item_ratings[item_id].append(rating)
                    all_ratings.append(rating)
        self.avg_global = statistics.mean(all_ratings) if all_ratings else 0
        self.item_stats = {}
        for item_id, votes in item_ratings.items():
            if len(votes) >= self.min_vots and item_id in self.dataset.items:
                self.item_stats[item_id] = {
                    'avg_item': statistics.mean(votes),
                    'num_vots': len(votes)
                }

    @abstractmethod
    def recommend(self, user_id, n=5):
        pass


class SimpleRecommender(Recommender):
    """Clase para un sistema de recomendación basado en popularidad."""
    def __init__(self, dataset, min_vots=10):
        super().__init__(dataset, min_vots)
        self._calculate_item_stats()

    def _calculate_score(self, avg_item, num_vots):
        if num_vots + self.min_vots == 0:
            return 0.0
        return (num_vots / (num_vots + self.min_vots)) * avg_item + \
            (self.min_vots / (num_vots + self.min_vots)) * self.avg_global

    def recommend(self, user_id, n=5, all_ratings=None):
        user_id = str(user_id)
        user_ratings = self.dataset.ratings.get(user_id, {})
        if all_ratings and user_id in all_ratings:
            user_ratings = {str(k): v for k, v in user_ratings.items()}
            user_ratings.update({str(k): v for k, v in all_ratings.get(user_id, {}).items()})

        unrated_items = [
            str(item_id) for item_id in self.item_stats
            if str(item_id) not in user_ratings or user_ratings.get(str(item_id), 0) == 0
        ]

        recommendations = []
        for item_id in unrated_items:
            stats = self.item_stats[item_id]
            score = self._calculate_score(stats['avg_item'], stats['num_vots'])
            recommendations.append((self.dataset.items[item_id], score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]


class CollaborativeRecommender(Recommender):
    """Clase para un sistema de recomendación colaborativo basado en usuarios."""
    def __init__(self, dataset, min_vots=10):
        super().__init__(dataset, min_vots)
        self.similarities = {}
        self.item_stats = self._calculate_item_stats_collab()
        self.global_mean = self._calculate_global_mean()

    def _calculate_item_stats_collab(self):
        item_ratings = defaultdict(list)
        for user_ratings in self.dataset.ratings.values():
            for item_id, rating in user_ratings.items():
                if rating > 0:
                    item_ratings[item_id].append(rating)
        item_stats = {}
        for item_id, votes in item_ratings.items():
            if len(votes) >= self.min_vots and item_id in self.dataset.items:
                item_stats[item_id] = {'num_vots': len(votes), 'mean': statistics.mean(votes)}
        return item_stats

    def _calculate_global_mean(self):
        all_ratings = []
        for user_ratings in self.dataset.ratings.values():
            all_ratings.extend([r for r in user_ratings.values() if r > 0])
        return statistics.mean(all_ratings) if all_ratings else 0.0

    def _calculate_similarities(self, user_id):
        user_id = str(user_id)
        if user_id in self.similarities:
            return self.similarities[user_id]

        similarities = {}
        user_ratings = self.dataset.ratings.get(user_id, {})
        if not user_ratings:
            Utils.log_message("WARNING", f"No hay valoraciones para el usuario {user_id}.")
            self.similarities[user_id] = similarities
            return similarities

        for other_user_id in self.dataset.ratings:
            if other_user_id != user_id:
                common_items = set(user_ratings.keys()) & set(self.dataset.ratings[other_user_id].keys())
                if not common_items:
                    continue

                user_vector = [user_ratings[item] for item in common_items]
                other_vector = [self.dataset.ratings[other_user_id][item] for item in common_items]

                dot_product = sum(u * v for u, v in zip(user_vector, other_vector))
                norm_user = math.sqrt(sum(u * u for u in user_vector))
                norm_other = math.sqrt(sum(v * v for v in other_vector))

                if norm_user == 0 or norm_other == 0:
                    continue

                similarity = dot_product / (norm_user * norm_other)
                similarity = max(min(similarity, 1.0), -1.0)
                similarities[other_user_id] = similarity

        self.similarities[user_id] = similarities
        Utils.log_message("INFO", f"Similitudes calculadas para {user_id}: 10 vecinos.")
        return similarities

    def predict_rating(self, user_id, item_id, user_mean, k=10, min_similarity=0.5):
        user_id = str(user_id)
        item_id = str(item_id)
        user_ratings = self.dataset.ratings.get(user_id, {})

        if item_id in user_ratings and user_ratings[item_id] > 0:
            return user_ratings[item_id]

        similarities = self._calculate_similarities(user_id)
        if not similarities:
            return None

        relevant_neighbors = [(other_id, sim) for other_id, sim in similarities.items()
                              if sim >= min_similarity and item_id in self.dataset.ratings.get(other_id, {})]
        sorted_neighbors = sorted(relevant_neighbors, key=lambda x: x[1], reverse=True)[:k]
        if not sorted_neighbors:
            return user_mean

        numerator = 0.0
        denominator = 0.0
        for other_user_id, similarity in sorted_neighbors:
            other_ratings = self.dataset.ratings.get(other_user_id, {})
            other_mean = statistics.mean(
                [r for r in other_ratings.values() if r > 0]) if other_ratings else self.global_mean
            diff = other_ratings[item_id] - other_mean
            numerator += similarity * diff
            denominator += abs(similarity)

        if denominator == 0:
            return user_mean

        prediction = user_mean + (numerator / denominator)
        prediction = max(min(prediction, 5.0), 0.5)
        return prediction

    def recommend(self, user_id, n=5, all_ratings=None, k=10):
        user_id = str(user_id)
        user_ratings = self.dataset.ratings.get(user_id, {})

        if all_ratings and user_id in all_ratings:
            user_ratings = {str(k): v for k, v in user_ratings.items()}
            user_ratings.update({str(k): v for k, v in all_ratings.get(user_id, {}).items()})

        user_mean = statistics.mean([r for r in user_ratings.values() if r > 0]) if user_ratings else self.global_mean

        unrated_items = [
            str(item_id) for item_id in self.item_stats
            if (str(item_id) not in user_ratings or user_ratings.get(str(item_id), 0) == 0)
        ]

        if not unrated_items:
            return []

        recommendations = []
        for item_id in unrated_items:
            score = self.predict_rating(user_id, item_id, user_mean, k)
            if score is not None:
                recommendations.append((item_id, score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:n]
        return [(self.dataset.items[item_id], score) for item_id, score in recommendations]


class ContentBasedRecommender(Recommender):
    """Clase para un sistema de recomendación basado en contenido (TF-IDF)."""
    def __init__(self, dataset, item_features, min_vots=3):
        super().__init__(dataset, min_vots)
        self._calculate_item_stats()
        self.item_features = item_features
        self._compute_tfidf_matrix()

    def _compute_tfidf_matrix(self):
        self.item_ids = list(self.item_features.keys())
        all_words = set()
        for feature_text in self.item_features.values():
            words = feature_text.split('|') if '|' in feature_text else feature_text.split()
            for word in words:
                all_words.add(word)
        self.vocab = list(all_words)
        self.vocab_size = len(self.vocab)

        self.tfidf_matrix = np.zeros((len(self.item_ids), self.vocab_size))
        doc_freq = np.zeros(self.vocab_size)

        for i, item_id in enumerate(self.item_ids):
            feature_text = self.item_features[item_id]
            words = feature_text.split('|') if '|' in feature_text else feature_text.split()
            total_words = len(words)
            word_counts = defaultdict(int)
            for word in words:
                if word in self.vocab:
                    word_idx = self.vocab.index(word)
                    word_counts[word_idx] += 1
            for word_idx, count in word_counts.items():
                self.tfidf_matrix[i, word_idx] = count / total_words if total_words > 0 else 0
                doc_freq[word_idx] += 1

        N = len(self.item_ids)
        idf = np.log(N / (doc_freq + 1))
        self.tfidf_matrix = self.tfidf_matrix * idf
        return self.tfidf_matrix

    def _compute_user_profile(self, user_ratings):
        rated_items = [item_id for item_id, rating in user_ratings.items() if rating > 0]
        if not rated_items:
            return None

        indices = []
        ratings = []
        for item_id in rated_items:
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                indices.append(idx)
                ratings.append(user_ratings[item_id])

        if not indices:
            return None

        ratings = np.array(ratings)
        tfidf_vectors = self.tfidf_matrix[indices]
        weighted_sum = np.sum(tfidf_vectors * ratings[:, np.newaxis], axis=0)
        total_weight = np.sum(ratings)
        if total_weight == 0:
            return None
        return weighted_sum / total_weight

    def _cosine_similarity(self, vector1, vector2):
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vector1, vector2) / (norm1 * norm2)

    def recommend(self, user_id, n=5):
        if user_id not in self.dataset.ratings:
            return []

        user_ratings = self.dataset.ratings[user_id]
        user_profile = self._compute_user_profile(user_ratings)
        if user_profile is None:
            return []

        unrated_items = [
            item_id for item_id in self.item_ids
            if item_id not in user_ratings or user_ratings[item_id] == 0
        ]
        if not unrated_items:
            return []

        recommendations = []
        p_max = 5.0
        for item_id in unrated_items:
            if item_id in self.dataset.items:
                idx = self.item_ids.index(item_id)
                item_vector = self.tfidf_matrix[idx]
                similarity = self._cosine_similarity(user_profile, item_vector)
                score = min(max(similarity * p_max, 0.5), 5.0)
                recommendations.append((self.dataset.items[item_id], score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]