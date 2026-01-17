from typing import List, Dict, Any, Optional
import networkx as nx
import json
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

from ..nlp.nlp_engine import NLPEngine


class KnowledgeGraphGenerator:
    def __init__(self, nlp_engine: NLPEngine):
        self.nlp_engine = nlp_engine
    
    def generate(
        self,
        text: str,
        max_nodes: int = 50,
        min_edge_weight: int = 1
    ) -> nx.Graph:
        logger.info("Generating knowledge graph")
        
        G = self.nlp_engine.build_concept_graph(text, top_n_concepts=max_nodes)
        
        edges_to_remove = [
            (u, v) for u, v, data in G.edges(data=True)
            if data.get('weight', 0) < min_edge_weight
        ]
        G.remove_edges_from(edges_to_remove)
        
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        self._add_centrality_measures(G)
        
        self._add_communities(G)
        
        logger.info(f"Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def _add_centrality_measures(self, G: nx.Graph):
        if G.number_of_nodes() == 0:
            return
        
        degree_centrality = nx.degree_centrality(G)
        for node, centrality in degree_centrality.items():
            G.nodes[node]['degree_centrality'] = centrality
        
        try:
            betweenness_centrality = nx.betweenness_centrality(G)
            for node, centrality in betweenness_centrality.items():
                G.nodes[node]['betweenness_centrality'] = centrality
        except:
            pass
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            for node, centrality in eigenvector_centrality.items():
                G.nodes[node]['eigenvector_centrality'] = centrality
        except:
            pass
    
    def _add_communities(self, G: nx.Graph):
        if G.number_of_nodes() == 0:
            return
        
        try:
            from community import community_louvain
            communities = community_louvain.best_partition(G)
            
            for node, community_id in communities.items():
                G.nodes[node]['community'] = community_id
        except ImportError:
            logger.warning("python-louvain not installed, skipping community detection")
    
    def export_to_json(self, G: nx.Graph, output_path: Path) -> Dict[str, Any]:
        data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'is_connected': nx.is_connected(G) if G.number_of_nodes() > 0 else False
            }
        }
        
        for node, attrs in G.nodes(data=True):
            node_data = {
                'id': node,
                'label': node,
                **attrs
            }
            data['nodes'].append(node_data)
        
        for u, v, attrs in G.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                **attrs
            }
            data['edges'].append(edge_data)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Knowledge graph exported to {output_path}")
        
        return data
    
    def visualize_matplotlib(
        self,
        G: nx.Graph,
        output_path: Path,
        figsize: tuple = (20, 16),
        node_size_scale: int = 3000
    ):
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, skipping visualization")
            return
        
        plt.figure(figsize=figsize)
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        node_sizes = [
            G.nodes[node].get('degree_centrality', 0.1) * node_size_scale
            for node in G.nodes()
        ]
        
        node_colors = [
            G.nodes[node].get('community', 0)
            for node in G.nodes()
        ]
        
        edge_widths = [
            G[u][v].get('weight', 1) * 0.5
            for u, v in G.edges()
        ]
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.3,
            edge_color='gray'
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.Set3,
            alpha=0.8
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title("Knowledge Graph Visualization", fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Knowledge graph visualization saved to {output_path}")
    
    def visualize_plotly(
        self,
        G: nx.Graph,
        output_path: Path
    ):
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, skipping visualization")
            return
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                )
            )
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node].get('degree_centrality', 0.1) * 50)
            node_color.append(G.nodes[node].get('community', 0))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title='Interactive Knowledge Graph',
                titlefont=dict(size=24),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        logger.info(f"Interactive knowledge graph saved to {output_path}")
    
    def extract_learning_paths(
        self,
        G: nx.Graph,
        start_concept: Optional[str] = None,
        max_paths: int = 5
    ) -> List[List[str]]:
        if G.number_of_nodes() == 0:
            return []
        
        if start_concept is None or start_concept not in G:
            centralities = nx.degree_centrality(G)
            start_concept = max(centralities, key=centralities.get)
        
        paths = []
        
        nodes_by_centrality = sorted(
            G.nodes(),
            key=lambda n: G.nodes[n].get('degree_centrality', 0),
            reverse=True
        )
        
        for target in nodes_by_centrality[:max_paths]:
            if target != start_concept:
                try:
                    path = nx.shortest_path(G, start_concept, target)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def get_graph_statistics(self, G: nx.Graph) -> Dict[str, Any]:
        if G.number_of_nodes() == 0:
            return {
                'node_count': 0,
                'edge_count': 0,
                'density': 0,
                'is_connected': False
            }
        
        stats = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
        }
        
        if nx.is_connected(G):
            stats['diameter'] = nx.diameter(G)
            stats['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        
        stats['average_clustering'] = nx.average_clustering(G)
        
        degree_centrality = nx.degree_centrality(G)
        stats['most_central_concepts'] = sorted(
            degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return stats
