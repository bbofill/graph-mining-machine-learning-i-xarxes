# -*- coding: utf-8 -*-
import math
import numpy as np
from src.utils import Utils


class ProtocolEvaluator:
    """
    Evaluador científico según el protocolo experimental de MovieLens (Secciones 5, 6, 7 y 10).
    Calcula NDCG@K (binario y gradual), Precision@K, Recall@K, MAE, RMSE e intervalos de confianza.
    """

    def __init__(self, dataset, k=10, relevance_threshold=4.0, non_relevant_max=2.0, exclude_ambiguous=True):
        self.dataset = dataset
        self.test_ratings = dataset.test_ratings
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.non_relevant_max = non_relevant_max
        self.exclude_ambiguous = exclude_ambiguous

    def get_user_ground_truth(self, user_id):
        """
        Aplica el criterio objetivo de relevancia (Sección 6 del Plan):
        - >= relevance_threshold (4.0★): Relevante (True)
        - <= non_relevant_max (2.0★): No Relevante (False)
        - (2.0★, 4.0★) -> 2.5 a 3.5★: Excluidas por ambiguas si exclude_ambiguous=True.
        """
        user_test = self.test_ratings.get(str(user_id), {})
        relevant_items = set()
        non_relevant_items = set()
        ambiguous_items = set()
        graded_relevance = {}

        for m_id, rating in user_test.items():
            if rating >= self.relevance_threshold:
                relevant_items.add(m_id)
                graded_relevance[m_id] = float(rating)
            elif rating <= self.non_relevant_max:
                non_relevant_items.add(m_id)
                graded_relevance[m_id] = 0.0
            else:
                ambiguous_items.add(m_id)
                if not self.exclude_ambiguous:
                    non_relevant_items.add(m_id)
                    graded_relevance[m_id] = 0.0

        return relevant_items, non_relevant_items, ambiguous_items, graded_relevance

    def evaluate_user(self, recommender, user_id, k=None):
        """
        Evalúa las recomendaciones Top-K para un único usuario siguiendo las fórmulas de la Sección 7.
        """
        k = k if k is not None else self.k
        relevant_items, _, ambiguous_items, graded_rel = self.get_user_ground_truth(user_id)

        # Si el usuario no tiene ningún ítem relevante en Test, no puede evaluarse en Recall@K/NDCG@K
        if len(relevant_items) == 0:
            return None

        # Pedimos un margen extra por si alguna recomendación cae en el conjunto ambiguo (2.5-3.5★) excluido
        raw_recs = recommender.recommend(user_id, n=k + len(ambiguous_items) + 5)

        filtered_recs = []
        for item_obj, score in raw_recs:
            m_id = item_obj.id
            if self.exclude_ambiguous and m_id in ambiguous_items:
                continue
            filtered_recs.append((m_id, score))
            if len(filtered_recs) >= k:
                break

        if not filtered_recs:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg_binary": 0.0,
                "ndcg_graded": 0.0
            }

        # 1. Precision@K y Recall@K
        hits = [1.0 if m_id in relevant_items else 0.0 for m_id, _ in filtered_recs]
        num_hits = sum(hits)
        precision_k = num_hits / float(k)
        recall_k = num_hits / float(len(relevant_items))

        # 2. NDCG@K Binario: DCG = sum_{i=1}^K hit_i / log2(i + 1)
        dcg_bin = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
        ideal_hits = min(k, len(relevant_items))
        idcg_bin = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
        ndcg_binary = (dcg_bin / idcg_bin) if idcg_bin > 0 else 0.0

        # 3. NDCG@K Gradual (basado en la valoración real de los ítems relevantes)
        gains = [graded_rel.get(m_id, 0.0) if m_id in relevant_items else 0.0 for m_id, _ in filtered_recs]
        dcg_grad = sum((2.0 ** g - 1.0) / math.log2(idx + 2) for idx, g in enumerate(gains))
        ideal_gains = sorted([graded_rel[m] for m in relevant_items], reverse=True)[:k]
        idcg_grad = sum((2.0 ** g - 1.0) / math.log2(idx + 2) for idx, g in enumerate(ideal_gains))
        ndcg_graded = (dcg_grad / idcg_grad) if idcg_grad > 0 else 0.0

        return {
            "precision": precision_k,
            "recall": recall_k,
            "ndcg_binary": ndcg_binary,
            "ndcg_graded": ndcg_graded
        }

    def evaluate_model_global(self, recommender, max_users=None, n_bootstrap=200):
        """
        Evalúa un modelo sobre todos los usuarios elegibles de Test y reporta la media global
        junto con intervalos de confianza al 95% (Bootstrap).
        """
        eligible_users = list(self.test_ratings.keys())
        if max_users and len(eligible_users) > max_users:
            eligible_users = eligible_users[:max_users]

        precisions, recalls, ndcgs_bin, ndcgs_grad = [], [], [], []
        mae_errors, rmse_sq_errors = [], []

        for u_id in eligible_users:
            res = self.evaluate_user(recommender, u_id, k=self.k)
            if res is not None:
                precisions.append(res["precision"])
                recalls.append(res["recall"])
                ndcgs_bin.append(res["ndcg_binary"])
                ndcgs_grad.append(res["ndcg_graded"])

            # Cálculo complementario de MAE / RMSE (compatibilidad con tu código original)
            u_test = self.test_ratings[u_id]
            top_preds = recommender.recommend(u_id, n=50)
            for item_obj, pred_score in top_preds:
                if item_obj.id in u_test and 0.5 <= pred_score <= 5.0:
                    real_r = u_test[item_obj.id]
                    mae_errors.append(abs(pred_score - real_r))
                    rmse_sq_errors.append((pred_score - real_r) ** 2)

        num_eval_users = len(ndcgs_bin)
        if num_eval_users == 0:
            return {}

        # Intervalo de confianza al 95% para NDCG@K mediante Bootstrap
        ndcg_arr = np.array(ndcgs_bin)
        boot_means = [
            np.mean(np.random.choice(ndcg_arr, size=num_eval_users, replace=True))
            for _ in range(n_bootstrap)
        ]
        ci_95 = 1.96 * np.std(boot_means)

        summary = {
            "Modelo": recommender.name,
            f"NDCG@{self.k}": float(np.mean(ndcgs_bin)),
            f"NDCG@{self.k}_CI95": float(ci_95),
            f"NDCG_Grad@{self.k}": float(np.mean(ndcgs_grad)),
            f"Precision@{self.k}": float(np.mean(precisions)),
            f"Recall@{self.k}": float(np.mean(recalls)),
            "MAE": float(np.mean(mae_errors)) if mae_errors else 0.0,
            "RMSE": float(np.sqrt(np.mean(rmse_sq_errors))) if rmse_sq_errors else 0.0,
            "Usuarios_Evaluados": num_eval_users,
            "Umbral_Relevancia": f">= {self.relevance_threshold}★"
        }

        Utils.log_message(
            "INFO",
            f"Evaluado [{recommender.name}] ({num_eval_users} usuarios) -> "
            f"NDCG@{self.k}: {summary[f'NDCG@{self.k}']:.4f} (±{ci_95:.4f}) | "
            f"Precision@{self.k}: {summary[f'Precision@{self.k}']:.4f} | "
            f"Recall@{self.k}: {summary[f'Recall@{self.k}']:.4f}"
        )
        return summary

    @staticmethod
    def verify_illustrative_example():
        """
        Verifica exactamente el Ejemplo Ilustrativo de la Sección 8 del Plan:
        Top-5 recomendado: Blade Runner (5★), Titanic (2★), Terminator (4.5★), Avatar (1.5★), Alien (4★).
        Con umbral >= 4★ y 4 películas relevantes en total en el test:
        Precision@5 = 3/5 = 0.60 y Recall@5 = 3/4 = 0.75.
        """
        top5 = [("Blade Runner", 5.0), ("Titanic", 2.0), ("Terminator", 4.5), ("Avatar", 1.5), ("Alien", 4.0)]
        threshold = 4.0
        total_relevant_in_test = 4
        hits = [1 if r >= threshold else 0 for _, r in top5]
        prec_5 = sum(hits) / 5.0
        rec_5 = sum(hits) / float(total_relevant_in_test)
        dcg_5 = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        idcg_5 = sum(1.0 / math.log2(i + 2) for i in range(total_relevant_in_test))
        ndcg_5 = dcg_5 / idcg_5
        return {"Precision@5": prec_5, "Recall@5": rec_5, "NDCG@5": ndcg_5}
