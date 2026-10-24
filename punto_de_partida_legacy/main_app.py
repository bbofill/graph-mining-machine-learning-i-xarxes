# -*- coding: utf-8 -*-
import csv
from io import StringIO
from collections import defaultdict

from utils import Utils
from dataset import Dataset
from recommenders import SimpleRecommender, CollaborativeRecommender, ContentBasedRecommender
from evaluation import Evaluator

def main():
    # Cargar datasets
    try:
        with open('ratings.csv', 'r', encoding='utf-8') as f:
            movies_ratings_csv = f.read()
        with open('Ratings_books.csv', 'r', encoding='utf-8') as f:
            books_ratings_csv = f.read()
    except FileNotFoundError as e:
        Utils.log_message("ERROR", f"No se encontró el archivo {e.filename}.")
        print(f"Error: No se encontró el archivo {e.filename}.")
        return

    # Cargar nombres de ítems
    movie_names = {}
    book_names = {}
    try:
        with open('movies.csv', 'r', encoding='utf-8') as f:
            movies_data = csv.reader(f)
            next(movies_data)
            for row in movies_data:
                movie_names[row[0]] = row[1]
        Utils.log_message("INFO", "Nombres de películas cargados.")
    except FileNotFoundError:
        print("Error: No se encontró movies.csv.")
        return

    try:
        with open('Books.csv', 'r', encoding='utf-8') as f:
            books_data = csv.reader(f)
            next(books_data)
            for row in books_data:
                book_names[row[0]] = row[1]
        Utils.log_message("INFO", "Nombres de libros cargados.")
    except FileNotFoundError:
        print("Error: No se encontró Books.csv.")
        return

    datasets = {
        'películas': None,
        'libros': Dataset('Libros', books_ratings_csv, 'User-ID', 'ISBN', 'Book-Rating', has_timestamp=False, item_names=book_names)
    }

    # Interacción inicial
    print("Conjuntos de datos disponibles: películas, libros")
    dataset_choice = input("Elija el conjunto de datos (películas/libros): ").lower()
    while dataset_choice not in datasets:
        dataset_choice = input("Inválido. Elija el conjunto de datos (películas/libros): ").lower()

    item_features = {}
    if dataset_choice == 'películas':
        try:
            movies_ids = set()
            temp_item_features = {}
            with open('movies.csv', 'r', encoding='utf-8') as f:
                movies_data = csv.reader(f)
                next(movies_data)
                for row in movies_data:
                    movie_id = row[0]
                    movies_ids.add(movie_id)
                    temp_item_features[movie_id] = row[2]

            ratings_ids = set()
            filtered_ratings_csv = []
            csv_file = StringIO(movies_ratings_csv)
            csvreader = csv.reader(csv_file)
            header = next(csvreader)
            filtered_ratings_csv.append(','.join(header))
            
            for row in csvreader:
                if len(row) < 4: continue
                movie_id = row[1]
                if movie_id in movies_ids:
                    ratings_ids.add(movie_id)
                    filtered_ratings_csv.append(','.join(row))
            
            filtered_ratings_csv = '\n'.join(filtered_ratings_csv)
            datasets['películas'] = Dataset('Películas', filtered_ratings_csv, 'userId', 'movieId', 'rating', has_timestamp=True, item_names=movie_names)

            for movie_id in (ratings_ids & movies_ids):
                item_features[movie_id] = temp_item_features[movie_id]
                
        except FileNotFoundError:
            print("Error: No se encontró movies.csv.")
            return
    else:
        with open('Books.csv', 'r', encoding='utf-8') as f:
            books_data = csv.reader(f)
            next(books_data)
            for row in books_data:
                item_features[row[0]] = row[1]

    dataset = datasets[dataset_choice]
    
    # Train-Test Split (80/20)
    train_ratings = defaultdict(dict)
    test_ratings = defaultdict(dict)
    for user_id, user_ratings in dataset.ratings.items():
        items = list(user_ratings.items())
        split_idx = int(len(items) * 0.8)
        
        for item_id, rating in items[:split_idx]: train_ratings[user_id][item_id] = rating
        for item_id, rating in items[split_idx:]: test_ratings[user_id][item_id] = rating
            
    dataset.ratings = train_ratings

    print("\nMétodos disponibles: simple, colaborativo, contenido")
    method_choice = input("Elija el método: ").lower()
    while method_choice not in ['simple', 'colaborativo', 'contenido']:
        method_choice = input("Inválido. Elija el método (simple/colaborativo/contenido): ").lower()

    if method_choice == 'simple': recommender = SimpleRecommender(dataset)
    elif method_choice == 'colaborativo': recommender = CollaborativeRecommender(dataset)
    elif method_choice == 'contenido': recommender = ContentBasedRecommender(dataset, item_features)

    evaluator = Evaluator(recommender, test_ratings)

    # CLI Loop
    while True:
        print("\nOpciones: 1. Hacer recomendación | 2. Evaluar | 3. Salir")
        choice = input("Elija (1/2/3): ")

        if choice == '1':
            user_id = input("ID del usuario: ")
            recommendations = recommender.recommend(user_id, n=5)
            if not recommendations:
                print(f"No hay recomendaciones para {user_id}.")
            else:
                print(f"\nTop 5 para {user_id} ({dataset.name}):")
                for i, (item, score) in enumerate(recommendations, 1):
                    print(f"{i}. {item.get_info()}, Puntuación: {score:.2f}")

        elif choice == '2':
            user_id = input("ID para evaluar: ")
            predictions, actual, mae, rmse = evaluator.evaluate(user_id, n=5)
            if not predictions:
                print(f"No hay predicciones en el set de prueba para {user_id}.")
            else:
                print(f"\nEvaluación {user_id}:")
                for i, (pred, act) in enumerate(zip(predictions, actual), 1):
                    print(f"{i}. Pred: {pred:.2f}, Real: {act:.2f}")
                print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        elif choice == '3':
            print("Saliendo del sistema...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()
