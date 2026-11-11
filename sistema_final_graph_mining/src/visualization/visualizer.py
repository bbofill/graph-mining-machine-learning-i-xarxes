# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.utils import Utils


class GraphAndResultsVisualizer:
    """
    Motor de visualización estática (PNG 300 DPI), topológica (GEXF)
    e interactiva autónoma (HTML5) para el sistema de recomendación.
    """

    def __init__(self, dataset, graph_builder):
        self.dataset = dataset
        self.builder = graph_builder
        Utils.setup_directories()

    def plot_bipartite_and_tripartite_subgraph(
        self,
        sample_users=8,
        include_genres=True,
        output_path="results/plots/grafo_estructura_movielens.png"
    ):
        """Dibuja el subgrafo multicapa Usuario - Película - Género en alta resolución."""
        G_full = self.builder.build_networkx_graph(include_genres=include_genres, min_rating_edge=4.0)
        selected_users = [f"U_{u}" for u in self.builder.user_ids[:sample_users]]
        sub_nodes = set(selected_users)

        for u_node in selected_users:
            if u_node in G_full:
                movies = [n for n in G_full.neighbors(u_node) if n.startswith("M_")][:4]
                sub_nodes.update(movies)
                if include_genres:
                    for m in movies:
                        genres = [g for g in G_full.neighbors(m) if g.startswith("G_")]
                        sub_nodes.update(genres)

        H = G_full.subgraph(sub_nodes).copy()

        users = [n for n, d in H.nodes(data=True) if d.get("node_type") == "user"]
        movies = [n for n, d in H.nodes(data=True) if d.get("node_type") == "movie"]
        genres = [n for n, d in H.nodes(data=True) if d.get("node_type") == "genre"]

        pos = {}
        for i, u in enumerate(users):
            pos[u] = (0.0, i * (1.0 / max(len(users), 1)))
        for i, m in enumerate(movies):
            pos[m] = (1.5, i * (1.0 / max(len(movies), 1)))
        for i, g in enumerate(genres):
            pos[g] = (3.0, i * (1.0 / max(len(genres), 1)))

        plt.figure(figsize=(14, 9), dpi=300)
        ax = plt.gca()
        ax.set_facecolor("#F8F9FA")

        # Aristas Usuario-Película y Película-Género
        edges_um = [(u, v) for u, v, d in H.edges(data=True) if d.get("edge_type") == "rated"]
        edges_mg = [(u, v) for u, v, d in H.edges(data=True) if d.get("edge_type") == "has_genre"]

        nx.draw_networkx_edges(H, pos, edgelist=edges_um, alpha=0.35, edge_color="#4EA8DE", width=1.2)
        if include_genres and edges_mg:
            nx.draw_networkx_edges(H, pos, edgelist=edges_mg, alpha=0.35, edge_color="#52B788", style="dashed", width=1.2)

        # Nodos por capa
        nx.draw_networkx_nodes(H, pos, nodelist=users, node_color="#1D3557", node_size=380, label="Usuarios (U)")
        nx.draw_networkx_nodes(H, pos, nodelist=movies, node_color="#F77F00", node_size=260, node_shape="s", label="Películas (M)")
        if include_genres and genres:
            nx.draw_networkx_nodes(H, pos, nodelist=genres, node_color="#2A9D8F", node_size=420, node_shape="D", label="Géneros (G)")

        labels = {}
        for n, d in H.nodes(data=True):
            if d.get("node_type") == "user":
                labels[n] = f"U:{d.get('raw_id')}"
            elif d.get("node_type") == "movie":
                t = d.get("title", n)
                labels[n] = t[:18] + "..." if len(t) > 18 else t
            else:
                labels[n] = d.get("title", n)

        nx.draw_networkx_labels(H, pos, labels=labels, font_size=7.5, font_weight="bold", font_color="#212529")

        plt.title("Arquitectura Topológica del Grafo Heterogéneo (Usuario – Película – Género)", fontsize=13, fontweight="bold", pad=14)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True)
        plt.axis("off")
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        Utils.log_message("INFO", f"Gráfico del subgrafo guardado en: {output_path}")

    def plot_recommendation_explanation_subgraph(
        self,
        user_id,
        movie_id,
        output_path="results/plots/grafo_explicacion.png"
    ):
        """Dibuja el subgrafo de caminos cortos (<=3 saltos) que explica por qué se recomienda un ítem."""
        G = self.builder.build_networkx_graph(include_genres=True, min_rating_edge=3.5)
        u_node = f"U_{user_id}"
        m_node = f"M_{movie_id}"

        if u_node not in G or m_node not in G:
            Utils.log_message("WARNING", f"No se encontraron nodos para explicar {u_node} -> {m_node}")
            return

        paths = []
        try:
            for p in nx.all_simple_paths(G, source=u_node, target=m_node, cutoff=3):
                paths.append(p)
                if len(paths) >= 12:
                    break
        except nx.NetworkXNoPath:
            pass

        sub_nodes = {u_node, m_node}
        for p in paths:
            sub_nodes.update(p)

        if len(sub_nodes) <= 2:
            neighbors_u = list(G.neighbors(u_node))[:4]
            neighbors_m = list(G.neighbors(m_node))[:4]
            sub_nodes.update(neighbors_u)
            sub_nodes.update(neighbors_m)

        H = G.subgraph(sub_nodes).copy()
        pos = nx.spring_layout(H, seed=42, k=0.9)

        plt.figure(figsize=(11, 7), dpi=300)
        node_colors = []
        node_sizes = []
        labels = {}

        for n, d in H.nodes(data=True):
            if n == u_node:
                node_colors.append("#E63946")
                node_sizes.append(650)
                labels[n] = f"Usuario Activo ({user_id})"
            elif n == m_node:
                node_colors.append("#FFB703")
                node_sizes.append(700)
                labels[n] = f"Recomendación:\n{d.get('title', n)[:22]}"
            elif d.get("node_type") == "user":
                node_colors.append("#457B9D")
                node_sizes.append(350)
                labels[n] = f"User {d.get('raw_id')}"
            elif d.get("node_type") == "movie":
                node_colors.append("#FB8500")
                node_sizes.append(380)
                labels[n] = d.get("title", n)[:18]
            else:
                node_colors.append("#2A9D8F")
                node_sizes.append(420)
                labels[n] = f"Género: {d.get('title', n)}"

        nx.draw_networkx_edges(H, pos, alpha=0.45, edge_color="#6C757D", width=1.6)
        nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, edgecolors="#212529", linewidths=1.2)
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, font_weight="bold")

        plt.title(f"Subgrafo de Explicabilidad Topológica: Usuario {user_id} → Película {movie_id}", fontsize=12, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        Utils.log_message("INFO", f"Subgrafo explicativo guardado en: {output_path}")

    def export_interactive_pyvis_html(
        self,
        sample_users=25,
        include_genres=True,
        output_html="results/graphs/grafo_interactivo_movielens.html"
    ):
        """
        Genera un archivo HTML5 interactivo autónomo (con buscador, filtros y zoom)
        que se puede abrir directamente en cualquier navegador sin errores de carga.
        """
        G_full = self.builder.build_networkx_graph(include_genres=include_genres, min_rating_edge=4.0)
        selected_users = [f"U_{u}" for u in self.builder.user_ids[:sample_users]]
        sub_nodes = set(selected_users)

        for u_node in selected_users:
            if u_node in G_full:
                movies = [n for n in G_full.neighbors(u_node) if n.startswith("M_")][:6]
                sub_nodes.update(movies)
                if include_genres:
                    for m in movies:
                        sub_nodes.update([g for g in G_full.neighbors(m) if g.startswith("G_")])

        H = G_full.subgraph(sub_nodes)

        nodes_data = []
        for node_id, data in H.nodes(data=True):
            ntype = data.get("node_type")
            degree = H.degree(node_id)
            if ntype == "user":
                label = f"User {data['raw_id']}"
                nodes_data.append({
                    "id": node_id,
                    "label": label,
                    "group": "user",
                    "title": f"Usuario ID: {data['raw_id']} ({degree} conexiones)",
                    "info": f"Nodo de Usuario | ID: {data['raw_id']} | Conexiones en subgrafo: {degree}",
                    "shape": "dot",
                    "size": 16 + min(degree * 1.5, 18),
                    "color": {"background": "#3B82F6", "border": "#93C5FD", "highlight": {"background": "#60A5FA", "border": "#FFFFFF"}}
                })
            elif ntype == "movie":
                full_title = data.get("title", node_id)
                genres_str = data.get("genres", "N/A")
                short_label = full_title[:24] + ("..." if len(full_title) > 24 else "")
                nodes_data.append({
                    "id": node_id,
                    "label": short_label,
                    "full_title": full_title,
                    "group": "movie",
                    "title": f"{full_title} ({genres_str})",
                    "info": f"Película: {full_title} | Géneros: {genres_str} | Conexiones: {degree}",
                    "shape": "star",
                    "size": 22 + min(degree * 1.5, 20),
                    "color": {"background": "#F59E0B", "border": "#FDE68A", "highlight": {"background": "#FBBF24", "border": "#FFFFFF"}}
                })
            else:
                g_title = data.get("title", node_id)
                nodes_data.append({
                    "id": node_id,
                    "label": g_title,
                    "group": "genre",
                    "title": f"Género: {g_title} ({degree} películas)",
                    "info": f"Género Cinematográfico: {g_title} | Películas conectadas: {degree}",
                    "shape": "diamond",
                    "size": 26 + min(degree * 1.2, 22),
                    "color": {"background": "#10B981", "border": "#A7F3D0", "highlight": {"background": "#34D399", "border": "#FFFFFF"}}
                })

        edges_data = []
        for u, v, edata in H.edges(data=True):
            if edata.get("edge_type") == "rated":
                w = float(edata.get("weight", 4.0))
                edges_data.append({
                    "from": u,
                    "to": v,
                    "value": w,
                    "title": f"Valoración: {w:.1f}★",
                    "color": {"color": "rgba(96, 165, 250, 0.35)", "highlight": "#38BDF8"}
                })
            else:
                edges_data.append({
                    "from": u,
                    "to": v,
                    "value": 1.2,
                    "title": "Pertenece al género",
                    "dashes": True,
                    "color": {"color": "rgba(52, 211, 153, 0.30)", "highlight": "#34D399"}
                })

        html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Explorador Topológico - MovieLens Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: 100%; background-color: #0E1117; color: #E6EDF3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; }}
  .app-wrapper {{ display: flex; flex-direction: column; width: 100%; height: 100vh; background: #0E1117; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 12px 18px; background: #161B22; border-bottom: 1px solid #30363D; z-index: 10; }}
  .brand {{ font-weight: 700; font-size: 14px; color: #58A6FF; margin-right: 10px; }}
  .search-input, .filter-select {{ background: #0D1117; color: #E6EDF3; border: 1px solid #30363D; border-radius: 6px; padding: 7px 12px; font-size: 13px; outline: none; }}
  .search-input {{ flex: 1; min-width: 220px; }}
  .search-input:focus, .filter-select:focus {{ border-color: #3B82F6; }}
  .btn {{ background: #21262D; color: #E6EDF3; border: 1px solid #30363D; border-radius: 6px; padding: 7px 12px; font-size: 13px; cursor: pointer; transition: 0.15s; }}
  .btn:hover {{ background: #30363D; border-color: #8B949E; }}
  .btn-primary {{ background: #2563EB; border-color: #3B82F6; color: #FFF; font-weight: 500; }}
  .btn-primary:hover {{ background: #1D4ED8; }}
  #canvas-container {{ position: relative; flex: 1; width: 100%; height: 100%; }}
  #mynetwork {{ width: 100%; height: 100%; }}
  .zoom-controls {{ position: absolute; bottom: 20px; right: 20px; display: flex; flex-direction: column; gap: 6px; z-index: 10; }}
  .zoom-btn {{ width: 38px; height: 38px; border-radius: 8px; background: rgba(22, 27, 34, 0.92); color: #E6EDF3; border: 1px solid #30363D; font-size: 18px; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
  .zoom-btn:hover {{ background: #2563EB; border-color: #60A5FA; }}
  .legend {{ position: absolute; bottom: 20px; left: 20px; background: rgba(22, 27, 34, 0.92); border: 1px solid #30363D; border-radius: 8px; padding: 10px 14px; font-size: 12px; display: flex; gap: 14px; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.4); z-index: 10; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .dot-user {{ width: 10px; height: 10px; border-radius: 50%; background: #3B82F6; display: inline-block; }}
  .dot-movie {{ width: 10px; height: 10px; border-radius: 2px; background: #F59E0B; display: inline-block; transform: rotate(45deg); }}
  .dot-genre {{ width: 10px; height: 10px; background: #10B981; display: inline-block; }}
  .info-panel {{ position: absolute; top: 16px; left: 16px; background: rgba(22, 27, 34, 0.95); border: 1px solid #30363D; border-radius: 8px; padding: 10px 14px; font-size: 13px; max-width: 440px; display: none; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
</style>
</head>
<body>
<div class="app-wrapper">
  <div class="toolbar">
    <span class="brand">MovieLens Graph Explorer</span>
    <input type="text" id="searchInput" list="nodeList" class="search-input" placeholder="Buscar película, usuario o género (ej. Drama, User 1)...">
    <datalist id="nodeList"></datalist>
    <button class="btn btn-primary" onclick="focusNode()">Localizar</button>
    <select id="groupFilter" class="filter-select" onchange="filterByGroup()">
      <option value="all">Mostrar: Todos los nodos ({len(nodes_data)})</option>
      <option value="movie">Solo Películas</option>
      <option value="user">Solo Usuarios</option>
      <option value="genre">Solo Géneros</option>
    </select>
    <button class="btn" id="physicsBtn" onclick="togglePhysics()">Pausar Físicas</button>
    <button class="btn" onclick="resetGraph()">Restablecer Vista</button>
  </div>
  <div id="canvas-container">
    <div id="mynetwork"></div>
    <div id="infoPanel" class="info-panel"></div>
    <div class="legend">
      <div class="legend-item"><span class="dot-user"></span> Usuario</div>
      <div class="legend-item"><span class="dot-movie"></span> Película (★)</div>
      <div class="legend-item"><span class="dot-genre"></span> Género (◆)</div>
    </div>
    <div class="zoom-controls">
      <button class="zoom-btn" title="Acercar Zoom" onclick="zoom(1.3)">+</button>
      <button class="zoom-btn" title="Alejar Zoom" onclick="zoom(0.75)">−</button>
      <button class="zoom-btn" title="Encajar en pantalla" onclick="resetGraph()">⊡</button>
    </div>
  </div>
</div>

<script>
  const rawNodes = {json.dumps(nodes_data)};
  const rawEdges = {json.dumps(edges_data)};

  const nodes = new vis.DataSet(rawNodes);
  const edges = new vis.DataSet(rawEdges);

  const datalist = document.getElementById('nodeList');
  rawNodes.forEach(n => {{
    const opt = document.createElement('option');
    opt.value = n.full_title || n.label;
    datalist.appendChild(opt);
  }});

  const container = document.getElementById('mynetwork');
  const options = {{
    nodes: {{ font: {{ color: '#E6EDF3', size: 13, strokeWidth: 3, strokeColor: '#0E1117' }} }},
    edges: {{ smooth: {{ type: 'continuous' }} }},
    physics: {{
      enabled: true,
      barnesHut: {{ gravitationalConstant: -3800, centralGravity: 0.25, springLength: 135, springConstant: 0.04, damping: 0.12 }},
      stabilization: {{ enabled: true, iterations: 120, fit: true }}
    }},
    interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true }}
  }};

  const network = new vis.Network(container, {{ nodes, edges }}, options);
  let physicsEnabled = true;

  network.on("click", function(params) {{
    const panel = document.getElementById('infoPanel');
    if (params.nodes.length > 0) {{
      const node = nodes.get(params.nodes[0]);
      panel.style.display = 'block';
      panel.innerHTML = '<b>' + (node.full_title || node.label) + '</b><br><span style="color:#8B949E">' + node.info + '</span>';
    }} else {{
      panel.style.display = 'none';
    }}
  }});

  function focusNode() {{
    const query = document.getElementById('searchInput').value.trim().toLowerCase();
    if (!query) return;
    const found = rawNodes.find(n => (n.full_title && n.full_title.toLowerCase().includes(query)) || n.label.toLowerCase().includes(query) || n.id.toLowerCase() === query);
    if (found) {{
      network.selectNodes([found.id]);
      network.focus(found.id, {{ scale: 1.6, animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
      const panel = document.getElementById('infoPanel');
      panel.style.display = 'block';
      panel.innerHTML = '<b>' + (found.full_title || found.label) + '</b><br><span style="color:#8B949E">' + found.info + '</span>';
    }}
  }}

  document.getElementById('searchInput').addEventListener('keydown', e => {{ if (e.key === 'Enter') focusNode(); }});

  function filterByGroup() {{
    const g = document.getElementById('groupFilter').value;
    if (g === 'all') {{
      nodes.update(rawNodes.map(n => ({{ id: n.id, hidden: false }})));
    }} else {{
      nodes.update(rawNodes.map(n => ({{ id: n.id, hidden: n.group !== g }})));
    }}
    network.fit({{ animation: true }});
  }}

  function togglePhysics() {{
    physicsEnabled = !physicsEnabled;
    network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
    document.getElementById('physicsBtn').innerText = physicsEnabled ? 'Pausar Físicas' : 'Reanudar Físicas';
  }}

  function zoom(factor) {{
    const scale = network.getScale() * factor;
    network.moveTo({{ scale: scale, animation: {{ duration: 250 }} }});
  }}

  function resetGraph() {{
    document.getElementById('searchInput').value = '';
    document.getElementById('groupFilter').value = 'all';
    document.getElementById('infoPanel').style.display = 'none';
    filterByGroup();
    network.unselectAll();
    network.fit({{ animation: {{ duration: 500 }} }});
  }}
</script>
</body>
</html>"""

        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_template)

        Utils.log_message("INFO", f"Grafo interactivo HTML generado en: {output_html}")

    def export_to_gephi(self, include_genres=True, output_gexf="results/graphs/movielens_graph.gexf"):
        """Exporta el grafo completo en formato GEXF para análisis externo en Gephi."""
        G = self.builder.build_networkx_graph(include_genres=include_genres, min_rating_edge=4.0)
        os.makedirs(os.path.dirname(output_gexf), exist_ok=True)
        nx.write_gexf(G, output_gexf)
        Utils.log_message("INFO", f"Archivo GEXF para Gephi exportado en: {output_gexf}")

    def plot_lightgcn_embeddings_tsne(
        self,
        lightgcn_model,
        max_items=450,
        output_path="results/plots/lightgcn_embeddings_tsne.png"
    ):
        """Proyecta en 2D mediante t-SNE los embeddings de películas aprendidos por LightGCN."""
        if not hasattr(lightgcn_model, "item_embeddings_final") or lightgcn_model.item_embeddings_final is None:
            Utils.log_message("WARNING", "El modelo LightGCN no contiene embeddings entrenados para t-SNE.")
            return

        item_embs = lightgcn_model.item_embeddings_final
        n_available = min(max_items, item_embs.shape[0])
        selected_indices = np.arange(n_available)

        sub_embs = item_embs[selected_indices]
        norms = np.linalg.norm(sub_embs, axis=1, keepdims=True) + 1e-10
        sub_embs_norm = sub_embs / norms

        genres_list = []
        for idx in selected_indices:
            m_id = self.builder.idx_to_item[idx]
            movie_obj = self.dataset.items.get(m_id)
            if movie_obj and movie_obj.genres:
                genres_list.append(movie_obj.genres[0])
            else:
                genres_list.append("Unknown")

        tsne = TSNE(n_components=2, perplexity=min(30, max(5, n_available // 4)), random_state=42, init="pca", learning_rate="auto")
        coords_2d = tsne.fit_transform(sub_embs_norm)

        unique_genres = sorted(list(set(genres_list)))[:10]
        cmap = plt.cm.get_cmap("tab10", len(unique_genres))

        plt.figure(figsize=(11, 8), dpi=300)
        for i, g_name in enumerate(unique_genres):
            mask = [g == g_name for g in genres_list]
            if any(mask):
                pts = coords_2d[mask]
                plt.scatter(pts[:, 0], pts[:, 1], label=g_name, alpha=0.8, s=48, color=cmap(i), edgecolors="none")

        plt.title("Proyección t-SNE de los Embeddings de Películas Aprendidos en el Grafo (LightGCN)", fontsize=12, fontweight="bold")
        plt.xlabel("Dimensión Latente t-SNE 1")
        plt.ylabel("Dimensión Latente t-SNE 2")
        plt.legend(title="Género Principal", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        Utils.log_message("INFO", f"Proyección t-SNE de embeddings guardada en: {output_path}")

    def plot_rq_comparison_chart(
        self,
        results_list,
        k=10,
        output_path="results/plots/comparativa_modelos_rq1_rq2.png"
    ):
        """Genera el gráfico de barras agrupadas comparando NDCG@K, Precision@K y Recall@K."""
        if not results_list:
            return

        model_names = [r["Modelo"] for r in results_list]
        ndcg_vals = [r[f"NDCG@{k}"] for r in results_list]
        prec_vals = [r[f"Precision@{k}"] for r in results_list]
        rec_vals = [r[f"Recall@{k}"] for r in results_list]

        x = np.arange(len(model_names))
        width = 0.25

        plt.figure(figsize=(13, 6.5), dpi=300)
        plt.bar(x - width, ndcg_vals, width, label=f"NDCG@{k} (Métrica Principal)", color="#1D3557")
        plt.bar(x, prec_vals, width, label=f"Precision@{k}", color="#457B9D")
        plt.bar(x + width, rec_vals, width, label=f"Recall@{k}", color="#E63946")

        plt.ylabel("Puntuación Media Offline", fontsize=11)
        plt.title(f"Comparativa Experimental de Modelos (Split Temporal Estricto - Top-{k})", fontsize=13, fontweight="bold")
        plt.xticks(x, model_names, rotation=20, ha="right", fontsize=9.5, fontweight="bold")
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        Utils.log_message("INFO", f"Gráfico comparativo de modelos guardado en: {output_path}")
