# -*- coding: utf-8 -*-
import os
import csv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import track

from src.utils import Utils
from src.data.dataset import MovieLensDataset
from src.graph.graph_builder import InteractionGraphBuilder
from src.models.recommenders import (
    PopularityRecommender,
    CollaborativeRecommender,
    MatrixFactorizationRecommender
)
from src.models.graph_models import PageRankGraphRecommender, LightGCNRecommender
from src.evaluation.evaluator import ProtocolEvaluator
from src.visualization.visualizer import GraphAndResultsVisualizer

console = Console()


def print_rich_results_table(results_list, k=10, title="Resultados del Experimento"):
    """Renderiza la tabla comparativa de métricas en la terminal usando Rich."""
    table = Table(title=title, show_header=True, header_style="bold magenta", border_style="cyan")

    table.add_column("Modelo", style="white", width=48)
    table.add_column(f"NDCG@{k}", justify="right", style="bold green")
    table.add_column(f"Precision@{k}", justify="right", style="blue")
    table.add_column(f"Recall@{k}", justify="right", style="yellow")

    for r in results_list:
        ndcg_val = r[f"NDCG@{k}"]
        ndcg_color = "[bold green]" if ndcg_val > 0.08 else "[white]"
        ndcg_str = f"{ndcg_color}{ndcg_val:.4f}[/] [dim]±{r[f'NDCG@{k}_CI95']:.4f}[/dim]"

        table.add_row(
            r["Modelo"],
            ndcg_str,
            f"{r[f'Precision@{k}']:.4f}",
            f"{r[f'Recall@{k}']:.4f}"
        )

    console.print(table)
    console.print("\n")


def export_results_to_files(results_list, k=10):
    """Guarda las métricas en CSV y en formato tabla Markdown dentro de results/logs/."""
    os.makedirs("results/logs", exist_ok=True)
    csv_path = "results/logs/metricas_experimento.csv"
    md_path = "results/logs/tabla_resultados.md"

    if not results_list:
        return

    # 1. Guardar CSV
    fieldnames = list(results_list[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    # 2. Guardar Tabla Markdown lista para el README de GitHub
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write(f"| Modelo | NDCG@{k} (±IC 95%) | Precision@{k} | Recall@{k} |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for r in results_list:
            f.write(
                f"| **{r['Modelo']}** | `{r[f'NDCG@{k}']:.4f} ±{r[f'NDCG@{k}_CI95']:.4f}` "
                f"| `{r[f'Precision@{k}']:.4f}` | `{r[f'Recall@{k}']:.4f}` |\n"
            )

    Utils.log_message("INFO", f"Métricas exportadas en: {csv_path} y {md_path}")


def run_full_experimental_protocol(dataset_version="ml-1m", max_eval_users=300):
    Utils.setup_directories()
    Utils.set_seed(42)

    with console.status("[bold cyan]Cargando dataset y construyendo grafos...", spinner="dots"):
        dataset = MovieLensDataset(version=dataset_version, min_user_ratings=10, min_item_ratings=5)
        dataset.temporal_train_test_split(train_ratio=0.8)
        graph_builder = InteractionGraphBuilder(dataset)

    models = [
        PopularityRecommender(dataset, min_vots=20),
        CollaborativeRecommender(dataset, mode="user", k=30),
        MatrixFactorizationRecommender(dataset, n_factors=40, n_epochs=12),
        PageRankGraphRecommender(dataset, graph_builder=graph_builder, include_genres=False, alpha=0.15),
        PageRankGraphRecommender(dataset, graph_builder=graph_builder, include_genres=True, alpha=0.15),
        LightGCNRecommender(dataset, graph_builder=graph_builder, emb_dim=32, n_layers=3, epochs=20)
    ]

    evaluator_main = ProtocolEvaluator(dataset, k=10, relevance_threshold=4.0, exclude_ambiguous=True)
    results_main = []

    for model in track(models, description="[green]Evaluando modelos..."):
        res = evaluator_main.evaluate_model_global(model, max_users=max_eval_users)
        if res:
            results_main.append(res)

    print_rich_results_table(results_main, k=10, title="[bold yellow]Resultados Principales (Relevancia >= 4.0★)")
    export_results_to_files(results_main, k=10)

    with console.status("[bold magenta]Generando todos los artefactos en results/...", spinner="dots"):
        visualizer = GraphAndResultsVisualizer(dataset, graph_builder)
        visualizer.plot_bipartite_and_tripartite_subgraph(sample_users=8, include_genres=True)
        visualizer.export_interactive_pyvis_html(sample_users=25, include_genres=True)
        visualizer.export_to_gephi(include_genres=True)
        visualizer.plot_lightgcn_embeddings_tsne(models[-1])
        visualizer.plot_rq_comparison_chart(results_main, k=10)

        # Generar automáticamente un subgrafo de explicabilidad de ejemplo en results/plots/
        sample_u = graph_builder.user_ids[0]
        top_rec = models[4].recommend(sample_u, n=1)
        if top_rec:
            visualizer.plot_recommendation_explanation_subgraph(sample_u, top_rec[0][0].id)

    console.print(Panel(
        "[bold green]¡Experimento completado con éxito![/bold green]\n"
        "Todos los resultados se han guardado en la carpeta [bold cyan]results/[/bold cyan]:\n"
        "  • [white]results/plots/comparativa_modelos_rq1_rq2.png[/white]\n"
        "  • [white]results/plots/lightgcn_embeddings_tsne.png[/white]\n"
        "  • [white]results/plots/grafo_estructura_movielens.png[/white]\n"
        "  • [white]results/plots/grafo_explicacion.png[/white]\n"
        "  • [white]results/graphs/grafo_interactivo_movielens.html[/white] (Ábrelo con doble clic)\n"
        "  • [white]results/graphs/movielens_graph.gexf[/white]\n"
        "  • [white]results/logs/metricas_experimento.csv[/white] y [white]tabla_resultados.md[/white]",
        expand=False,
        border_style="green"
    ))


def interactive_cli():
    Utils.setup_directories()
    console.print(Panel.fit(
        "[bold cyan]GRAPH-BASED MOVIE RECOMMENDATION SYSTEM[/bold cyan]\n[dim]MovieLens 1M / Small[/dim]",
        border_style="blue"
    ))

    dataset_choice = Prompt.ask("Seleccione dataset (1: MovieLens 1M / 2: Small)", choices=["1", "2"], default="1")
    dataset_version = "ml-1m" if dataset_choice == "1" else "ml-latest-small"

    mode = Prompt.ask("Seleccione modo (A: Experimento Completo / B: Interactivo)", choices=["A", "B"], default="A")

    if mode == "A":
        max_u = IntPrompt.ask("Número máximo de usuarios a evaluar", default=300)
        run_full_experimental_protocol(dataset_version=dataset_version, max_eval_users=max_u)
        return

    with console.status("[bold cyan]Inicializando entorno interactivo..."):
        Utils.set_seed(42)
        dataset = MovieLensDataset(version=dataset_version)
        dataset.temporal_train_test_split(train_ratio=0.8)
        builder = InteractionGraphBuilder(dataset)
        visualizer = GraphAndResultsVisualizer(dataset, builder)

    console.print("\n[bold]Modelos disponibles:[/bold]")
    console.print("  [cyan]1.[/cyan] Popularidad (Baseline)")
    console.print("  [cyan]2.[/cyan] PageRank sobre Grafo Tripartito")
    console.print("  [cyan]3.[/cyan] LightGCN (Graph Neural Network)")

    m_choice = Prompt.ask("Elija el modelo", choices=["1", "2", "3"], default="2")

    models_dict = {
        "1": lambda: PopularityRecommender(dataset),
        "2": lambda: PageRankGraphRecommender(dataset, builder, include_genres=True),
        "3": lambda: LightGCNRecommender(dataset, builder)
    }

    with console.status("[bold green]Entrenando modelo seleccionado..."):
        recommender = models_dict[m_choice]()

    while True:
        console.print("\n[bold yellow]Opciones:[/bold yellow]")
        console.print("  [1] Recomendar Top-10 a usuario")
        console.print("  [2] Explicar Recomendación (Generar Grafo en results/plots/)")
        console.print("  [3] Salir")

        action = Prompt.ask("Elija una opción", choices=["1", "2", "3"])

        if action == "1":
            u_id = Prompt.ask("[bold cyan]ID de usuario[/bold cyan]")
            recs = recommender.recommend(u_id, n=10)

            rec_table = Table(title=f"Top 10 para Usuario {u_id}", style="cyan")
            rec_table.add_column("Rank", justify="center", style="bold")
            rec_table.add_column("Película", style="white")
            rec_table.add_column("Score", justify="right", style="green")

            for i, (movie, score) in enumerate(recs, 1):
                rec_table.add_row(str(i), movie.get_info(), f"{score:.4f}")
            console.print(rec_table)

        elif action == "2":
            u_id = Prompt.ask("[bold cyan]ID de usuario[/bold cyan]")
            recs = recommender.recommend(u_id, n=1)
            if recs:
                movie_id = recs[0][0].id
                with console.status(f"[bold magenta]Generando grafo explicativo para '{recs[0][0].name}'..."):
                    visualizer.plot_recommendation_explanation_subgraph(u_id, movie_id)
                console.print("[bold green]¡Grafo exportado en results/plots/grafo_explicacion.png![/bold green]")

        elif action == "3":
            console.print("[bold red]Saliendo...[/bold red]")
            break


if __name__ == "__main__":
    interactive_cli()
