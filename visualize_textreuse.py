import pandas as pd # type: ignore
import networkx as nx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from collections import defaultdict
import numpy as np
import os

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

def create_publication_network(reuse_df, weight_by='count', output_dir='reuse_visualizations'):
    """
    Create network where nodes are publications and edges are text reuse instances
    
    Parameters:
    - reuse_df: filtered reuse results
    - weight_by: 'count' (number of matches), 'total_words' (sum of match lengths),
                 or 'avg_similarity' (average similarity score)
    - output_dir: directory to save visualization
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
    
    # Visualize
    visualize_publication_network(G, output_file=os.path.join(output_dir, f'network_publications_{weight_by}.png'))
    
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
        try:
            pub = metadata.loc[metadata['page_id'] == node, 'publication_name'].iloc[0]
            G.nodes[node]['publication'] = pub
        except (IndexError, KeyError):
            # Handle case where page_id is not found in metadata
            G.nodes[node]['publication'] = f"Unknown_Publication_{node}"
    
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
    
    # Labels for all nodes showing page_id
    node_labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, 
                          font_size=8, font_weight='bold', ax=ax)
    
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

def create_page_level_heatmap(reuse_df, output_dir='reuse_visualizations'):
    """
    Create heatmap showing text reuse at the page level
    Shows which specific pages share text with other pages
    """
    # Load metadata to get page information
    metadata = pd.read_csv('page_metadata.csv')
    
    # Use the correct column names
    source_col = 'source_page_id'
    target_col = 'target_page_id'
    
    print(f"Using columns: {source_col} and {target_col}")
    
    # Merge to get source and target page information
    reuse_with_pages = reuse_df.merge(
        metadata[['page_id', 'publication_name', 'filename']], 
        left_on=source_col, 
        right_on='page_id',
        how='left'
    ).drop(columns=['page_id']).rename(columns={
        'publication_name': 'source_pub', 
        'filename': 'source_file'
    })
    
    reuse_with_pages = reuse_with_pages.merge(
        metadata[['page_id', 'publication_name', 'filename']], 
        left_on=target_col, 
        right_on='page_id',
        how='left'
    ).drop(columns=['page_id']).rename(columns={
        'publication_name': 'target_pub', 
        'filename': 'target_file'
    })

    
    # Create page labels (publication/filename)
    reuse_with_pages['source_label'] = reuse_with_pages['source_pub'] + '/' + reuse_with_pages['source_file'].apply(
        lambda x: x.split('/')[-1].replace('.txt', '') if isinstance(x, str) else 'unknown'
    )
    reuse_with_pages['target_label'] = reuse_with_pages['target_pub'] + '/' + reuse_with_pages['target_file'].apply(
        lambda x: x.split('/')[-1].replace('.txt', '') if isinstance(x, str) else 'unknown'
    )
    
    # Create adjacency matrix
    page_pairs = reuse_with_pages.groupby(['source_label', 'target_label']).size().reset_index(name='count')
    
    # Get all unique pages
    all_pages = sorted(set(page_pairs['source_label'].unique()) | set(page_pairs['target_label'].unique()))
    
    print(f"Total unique pages: {len(all_pages)}")
    
    if len(all_pages) > 200:
        print(f"⚠️  Warning: {len(all_pages)} pages will create a very large heatmap")
        print("Consider using create_top_pages_heatmap() instead")
    
    # Create matrix
    matrix = pd.DataFrame(0, index=all_pages, columns=all_pages)
    
    for _, row in page_pairs.iterrows():
        matrix.loc[row['source_label'], row['target_label']] = row['count']
        matrix.loc[row['target_label'], row['source_label']] = row['count']  # Make symmetric
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Use log scale for better visualization if there's high variance
    plot_data = np.log1p(matrix)  # log(1 + x) to handle zeros
    
    sns.heatmap(plot_data, 
                cmap='YlOrRd',
                cbar_kws={'label': 'Log(1 + Match Count)'},
                square=True,
                ax=ax)
    
    plt.title('Page-Level Text Reuse Heatmap', fontsize=16, pad=20)
    plt.xlabel('Target Page', fontsize=12)
    plt.ylabel('Source Page', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'page_level_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Also create a simplified version grouped by publication
    print("\nCreating summary by publication...")
    
    # Extract publication from label
    reuse_with_pages['source_pub_only'] = reuse_with_pages['source_label'].apply(lambda x: x.split('/')[0])
    reuse_with_pages['target_pub_only'] = reuse_with_pages['target_label'].apply(lambda x: x.split('/')[0])
    
    pub_summary = reuse_with_pages.groupby(['source_pub_only', 'target_pub_only']).size().reset_index(name='total_matches')
    
    print("\nTotal matches between publications:")
    print(pub_summary.pivot(index='source_pub_only', columns='target_pub_only', values='total_matches').fillna(0))

def create_top_pages_heatmap(reuse_df, top_n=50, output_dir='reuse_visualizations'):
    """
    Create heatmap for only the top N most connected pages
    More readable than full page-level heatmap
    """
    # Load metadata
    metadata = pd.read_csv('page_metadata.csv')
    
    # Use the correct column names
    source_col = 'source_page_id'
    target_col = 'target_page_id'
    
    print(f"Using columns: {source_col} and {target_col}")
    
    # Merge page information using page_id
    reuse_with_pages = reuse_df.merge(
        metadata[['page_id', 'publication_name', 'filename']], 
        left_on=source_col, 
        right_on='page_id',
        how='left'
    ).drop(columns=['page_id']).rename(columns={
        'publication_name': 'source_pub', 
        'filename': 'source_file'
    })
    
    reuse_with_pages = reuse_with_pages.merge(
        metadata[['page_id', 'publication_name', 'filename']], 
        left_on=target_col, 
        right_on='page_id',
        how='left'
    ).drop(columns=['page_id']).rename(columns={
        'publication_name': 'target_pub', 
        'filename': 'target_file'
    })
    
    # Create simplified labels (just filename without path)
    reuse_with_pages['source_label'] = reuse_with_pages['source_file'].apply(
        lambda x: x.split('/')[-1].replace('.txt', '') if isinstance(x, str) else 'unknown'
    )
    reuse_with_pages['target_label'] = reuse_with_pages['target_file'].apply(
        lambda x: x.split('/')[-1].replace('.txt', '') if isinstance(x, str) else 'unknown'
    )
    
    # Count connections per page
    source_counts = reuse_with_pages['source_label'].value_counts()
    target_counts = reuse_with_pages['target_label'].value_counts()
    total_counts = (source_counts + target_counts).fillna(0).sort_values(ascending=False)
    
    # Get top N pages
    top_pages = total_counts.head(top_n).index.tolist()
    
    print(f"\nTop {top_n} most connected pages:")
    for i, page in enumerate(top_pages[:10], 1):
        print(f"  {i}. {page}: {int(total_counts[page])} connections")
    
    # Filter data to top pages
    filtered = reuse_with_pages[
        reuse_with_pages['source_label'].isin(top_pages) & 
        reuse_with_pages['target_label'].isin(top_pages)
    ].copy()
    
    if len(filtered) == 0:
        print("⚠️  No data to plot after filtering")
        return
    
    # Create matrix
    page_pairs = filtered.groupby(['source_label', 'target_label']).size().reset_index(name='count')
    matrix = pd.DataFrame(0, index=top_pages, columns=top_pages)
    
    for _, row in page_pairs.iterrows():
        matrix.loc[row['source_label'], row['target_label']] = row['count']
        matrix.loc[row['target_label'], row['source_label']] = row['count']
    
    # Visualize
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(matrix, 
                cmap='YlOrRd',
                cbar_kws={'label': 'Match Count'},
                square=True,
                annot=False,
                fmt='g',
                ax=ax)
    
    plt.title(f'Top {top_n} Most Connected Pages - Text Reuse Heatmap', fontsize=14, pad=20)
    plt.xlabel('Target Page', fontsize=10)
    plt.ylabel('Source Page', fontsize=10)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'top_{top_n}_pages_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def create_temporal_visualization(reuse_df, output_dir='reuse_visualizations'):
    """
    Show text reuse over time
    """
    # Parse dates with correct format (YYYY-MM-DD)
    reuse_df['source_date'] = pd.to_datetime(reuse_df['source_date'], format='%Y-%m-%d', errors='coerce')
    reuse_df['target_date'] = pd.to_datetime(reuse_df['target_date'], format='%Y-%m-%d', errors='coerce')
    
    # Check for any invalid dates
    invalid_dates = reuse_df['source_date'].isna().sum() + reuse_df['target_date'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: {invalid_dates} invalid dates found and excluded")
    
    # Remove rows with invalid dates
    reuse_df = reuse_df.dropna(subset=['source_date', 'target_date'])
    
    if len(reuse_df) == 0:
        print("No valid dates found for temporal visualization")
        return
    
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
    output_file = os.path.join(output_dir, 'temporal_reuse.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved temporal visualization to {output_file}")
    plt.close()

def main():
    # Create output directory
    output_dir = 'reuse_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Load data
    print("Loading text reuse data...")
    reuse_df = load_filtered_reuse_data()

     # Load metadata for page network
    print("Loading metadata...")
    metadata = pd.read_csv('page_metadata.csv')
    
    # Create all visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    # 1. Publication-level network
    print("\n1. Publication Network...")
    create_publication_network(reuse_df, weight_by='count', output_dir=output_dir)
    
    # 2. Page-level network
    print("\n2. Page Network...")
    page_network = create_page_network(reuse_df, metadata, min_connections=1)
    visualize_page_network(page_network, output_file=os.path.join(output_dir, 'page_network.png'), 
                          title='Text Reuse Network (Page Level)')
    
    # 3. Timeline analysis
    print("\n3. Timeline Analysis...")
    create_temporal_visualization(reuse_df, output_dir=output_dir)
    
    # 4. Page-level heatmaps
    print("\n" + "="*80)
    print("4. Creating page-level heatmaps...")
    print("="*80)
    
    create_top_pages_heatmap(reuse_df, top_n=50, output_dir=output_dir)  # Top 50 pages
    create_page_level_heatmap(reuse_df, output_dir=output_dir)  # Full heatmap (may be very large)
    
    print("\n" + "="*80)
    print("✅ All visualizations complete!")
    print("="*80)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - network_publications_count.png")
    print("  - page_network.png")
    print("  - temporal_reuse.png")
    print("  - top_50_pages_heatmap.png")
    print("  - page_level_heatmap.png")
    print("="*80)


if __name__ == "__main__":
    main()