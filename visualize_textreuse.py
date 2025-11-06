import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_filtered_reuse_data(filepath='text_reuse_results.csv'):
    """Load FULL reuse data with citation filtering only"""
    df = pd.read_csv(filepath)
    
    # Filter citations only
    citation_keywords = ['archives of sexuality and gender', 'gale', 'cengage']
    is_citation = df['matched_text'].apply(
        lambda x: any(kw in x.lower() for kw in citation_keywords)
    )
    
    # Keep everything except citations
    filtered = df[~is_citation].copy()
    
    print(f"Loaded {len(filtered)} matches for visualization")
    print(f"Removed {is_citation.sum()} citation matches")
    return filtered

def create_publication_network(reuse_df, weight_by='count'):
    """
    Create network where nodes are publications and edges are text reuse instances
    
    Parameters:
    - reuse_df: filtered reuse results
    - weight_by: 'count' (number of matches), 'total_words' (sum of match lengths),
                 or 'avg_similarity' (average similarity score)
    """
    # Aggregate by publication pair
    agg_dict = {
        'match_length_words': ['sum', 'mean', 'count'],
        'similarity_score': 'mean'
    }
    
    pub_connections = reuse_df.groupby(
        ['source_publication', 'target_publication']
    ).agg(agg_dict).reset_index()
    
    # Flatten column names
    pub_connections.columns = [
        'source', 'target', 'total_words', 'avg_words', 'count', 'avg_similarity'
    ]
    
    # Create graph
    G = nx.DiGraph()
    
    # Add edges with attributes
    for _, row in pub_connections.iterrows():
        if weight_by == 'count':
            weight = row['count']
        elif weight_by == 'total_words':
            weight = row['total_words']
        elif weight_by == 'avg_similarity':
            weight = row['avg_similarity']
        else:
            weight = row['count']
        
        G.add_edge(
            row['source'], 
            row['target'],
            weight=weight,
            count=row['count'],
            total_words=row['total_words'],
            avg_words=row['avg_words'],
            avg_similarity=row['avg_similarity']
        )
    
    return G, pub_connections

def visualize_publication_network(G, output_file='network_publications.png', 
                                  title='Text Reuse Network Across Publications'):
    """
    Visualize publication-level network
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on degree (how connected they are)
    node_sizes = [G.degree(node) * 1500 for node in G.nodes()]
    
    # Edge widths based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [3 * (w / max_weight) for w in edge_weights]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.9, ax=ax)
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.6, edge_color='gray',
                          arrows=True, arrowsize=20, 
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Add edge labels for counts
    edge_labels = {(u, v): f"{G[u][v]['count']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved publication network to {output_file}")
    plt.close()

def create_page_network(reuse_df, metadata, min_connections=1):
    """
    Create network where nodes are individual pages
    
    Parameters:
    - reuse_df: filtered reuse results
    - metadata: page metadata with publication info
    - min_connections: minimum number of connections for a page to be included
    """
    G = nx.DiGraph()
    
    # Add edges between pages
    for _, row in reuse_df.iterrows():
        G.add_edge(
            row['source_page_id'],
            row['target_page_id'],
            weight=row['match_length_words'],
            similarity=row['similarity_score'],
            match_text=row['matched_text'][:100]  # Preview
        )
    
    # Filter nodes with minimum connections
    if min_connections > 1:
        nodes_to_keep = [node for node in G.nodes() if G.degree(node) >= min_connections]
        G = G.subgraph(nodes_to_keep).copy()
    
    # Add publication attribute to nodes from metadata
    for node in G.nodes():
        pub = metadata.loc[metadata['page_id'] == node, 'publication'].iloc[0]
        G.nodes[node]['publication'] = pub
    
    return G

def visualize_page_network(G, output_file='network_pages.png',
                          title='Text Reuse Network (Page Level)'):
    """
    Visualize page-level network with color-coding by publication
    """
    if len(G.nodes()) == 0:
        print("No pages to visualize with current filters")
        return
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Color by publication
    publications = list(set(nx.get_node_attributes(G, 'publication').values()))
    color_map = {pub: plt.cm.tab10(i) for i, pub in enumerate(publications)}
    node_colors = [color_map[G.nodes[node]['publication']] for node in G.nodes()]
    
    # Node sizes
    node_sizes = [G.degree(node) * 300 + 100 for node in G.nodes()]
    
    # Edge widths
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 * (w / max_weight) for w in edge_weights]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color=node_colors, alpha=0.8, ax=ax)
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.4, edge_color='gray',
                          arrows=True, arrowsize=10,
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Labels only for highly connected nodes
    high_degree_nodes = {node: node for node in G.nodes() if G.degree(node) > 2}
    nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, 
                          font_size=8, ax=ax)
    
    # Legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_map[pub], markersize=10,
                                 label=pub) for pub in publications]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved page network to {output_file}")
    plt.close()

def create_connection_matrix(reuse_df):
    """
    Create a heatmap showing connection strength between publications
    """
    # Aggregate by publication pair
    matrix_data = reuse_df.groupby(
        ['source_publication', 'target_publication']
    ).size().reset_index(name='count')
    
    # Pivot to matrix
    matrix = matrix_data.pivot(
        index='source_publication',
        columns='target_publication',
        values='count'
    ).fillna(0)
    
    return matrix

def visualize_connection_matrix(matrix, output_file='heatmap_connections.png'):
    """
    Visualize connection matrix as heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Text Reuse Instances'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('Text Reuse Connections Between Publications', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Publication', fontsize=12)
    ax.set_ylabel('Source Publication', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved connection matrix to {output_file}")
    plt.close()

def create_temporal_visualization(reuse_df, output_file='temporal_reuse.png'):
    """
    Show text reuse over time
    """
    # Ensure dates are datetime
    reuse_df['source_date'] = pd.to_datetime(reuse_df['source_date'])
    reuse_df['target_date'] = pd.to_datetime(reuse_df['target_date'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Timeline of reuse instances
    for _, row in reuse_df.iterrows():
        ax1.plot([row['source_date'], row['target_date']], [0, 1], 
                alpha=0.3, color='steelblue')
        ax1.scatter([row['source_date']], [0], alpha=0.6, s=50, color='green')
        ax1.scatter([row['target_date']], [1], alpha=0.6, s=50, color='red')
    
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Source', 'Target'])
    ax1.set_title('Text Reuse Flow Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency over time
    reuse_df['month'] = reuse_df['target_date'].dt.to_period('M')
    monthly_counts = reuse_df.groupby('month').size()
    
    ax2.bar(range(len(monthly_counts)), monthly_counts.values, 
           color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(monthly_counts)))
    ax2.set_xticklabels([str(m) for m in monthly_counts.index], rotation=45)
    ax2.set_title('Text Reuse Instances Per Month', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Instances', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved temporal visualization to {output_file}")
    plt.close()

def generate_network_statistics(G, pub_connections):
    """
    Calculate and print network statistics
    """
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    
    print(f"\nNodes (Publications): {G.number_of_nodes()}")
    print(f"Edges (Connections): {G.number_of_edges()}")
    print(f"Network Density: {nx.density(G):.3f}")
    
    print("\n--- Degree Centrality (Most Connected Publications) ---")
    degree_cent = nx.degree_centrality(G)
    for pub, cent in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True):
        print(f"{pub}: {cent:.3f}")
    
    print("\n--- In-Degree (Publications Receiving Most Content) ---")
    in_degree = dict(G.in_degree())
    for pub, deg in sorted(in_degree.items(), key=lambda x: x[1], reverse=True):
        print(f"{pub}: {deg} incoming connections")
    
    print("\n--- Out-Degree (Publications Sharing Most Content) ---")
    out_degree = dict(G.out_degree())
    for pub, deg in sorted(out_degree.items(), key=lambda x: x[1], reverse=True):
        print(f"{pub}: {deg} outgoing connections")
    
    print("\n--- Strongest Connections (by number of matches) ---")
    top_connections = pub_connections.nlargest(10, 'count')
    for _, row in top_connections.iterrows():
        print(f"{row['source']} → {row['target']}: {row['count']} matches "
              f"({row['total_words']} total words)")
    
    # Check for reciprocity
    if nx.is_directed(G):
        print(f"\nReciprocity: {nx.reciprocity(G):.3f}")
        print("(Proportion of bidirectional connections)")

def main():
    """
    Main execution function
    """
    print("Starting Text Reuse Network Visualization\n")
    
    # 1. Load data
    reuse_data = load_filtered_reuse_data('text_reuse_results.csv')
    
    if len(reuse_data) == 0:
        print("No data to visualize!")
        return
    
    # 2. Create publication-level network
    print("\nCreating publication-level network...")
    G_pub, pub_connections = create_publication_network(reuse_data, weight_by='count')
    
    # 3. Generate statistics
    generate_network_statistics(G_pub, pub_connections)
    
    # 4. Visualize publication network
    print("\nGenerating visualizations...")
    visualize_publication_network(G_pub)
    
    # 5. Create and visualize connection matrix
    matrix = create_connection_matrix(reuse_data)
    visualize_connection_matrix(matrix)
    
    # 6. Temporal visualization
    create_temporal_visualization(reuse_data)
    
    # 7. Page-level network (if you want it)
    # Uncomment if you have metadata loaded and want page-level viz
    # metadata = pd.read_csv('metadata.csv')
    # G_pages = create_page_network(reuse_data, metadata, min_connections=1)
    # visualize_page_network(G_pages)
    
    print("\n" + "="*60)
    print("Visualization complete! Check output files:")
    print("  - network_publications.png")
    print("  - heatmap_connections.png")
    print("  - temporal_reuse.png")
    print("="*60)

if __name__ == "__main__":
    main()