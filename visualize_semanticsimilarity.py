"""
Semantic Similarity Visualization Script
Works from all_page_similarities.csv with citation filtering
Generates PAGE-LEVEL heatmap and clustering visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path('semantic_visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_filtered_similarity_data(filepath='semantic_results/all_page_similarities.csv'):
    """
    Load page similarity data with citation filtering
    Filters based on text previews containing citation markers
    """
    print("Loading page similarity data...")
    df = pd.read_csv(filepath)
    
    print(f"Original similarity pairs: {len(df)}")
    
    # Citation keywords to filter
    citation_keywords = [
        'archives of sexuality and gender',
        'gale',
        'cengage',
        'primary sources'
    ]
    
    # Check if either text preview contains citation language
    def contains_citation(row):
        """Check if either text preview contains citation markers"""
        text1 = str(row['page1_text_preview']).lower()
        text2 = str(row['page2_text_preview']).lower()
        
        return any(kw in text1 or kw in text2 for kw in citation_keywords)
    
    is_citation = df.apply(contains_citation, axis=1)
    
    # Keep everything except citations
    filtered = df[~is_citation].copy()
    
    print(f"After filtering citations: {len(filtered)}")
    print(f"Removed {is_citation.sum()} citation matches")
    
    # Save filtered data
    filtered.to_csv(OUTPUT_DIR / 'page_similarities_no_citations.csv', index=False)
    print(f"Saved filtered data to: {OUTPUT_DIR / 'page_similarities_no_citations.csv'}")
    
    return filtered

def create_similarity_matrix_from_pairs(similarity_df):
    """
    Convert pairwise similarity dataframe to full similarity matrix
    Returns matrix and list of page_ids in order
    """
    print("\nCreating full similarity matrix from pairs...")
    
    # Get all unique page IDs
    all_pages = sorted(set(similarity_df['page1_id'].unique()) | 
                      set(similarity_df['page2_id'].unique()))
    n_pages = len(all_pages)
    
    print(f"Total pages: {n_pages}")
    
    # Create page_id to index mapping
    page_to_idx = {page: i for i, page in enumerate(all_pages)}
    
    # Initialize similarity matrix (with 1s on diagonal)
    similarity_matrix = np.eye(n_pages)
    
    # Fill in the similarity values
    for _, row in similarity_df.iterrows():
        i = page_to_idx[row['page1_id']]
        j = page_to_idx[row['page2_id']]
        sim = row['similarity']
        
        # Fill both directions (symmetric)
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return similarity_matrix, all_pages

def get_page_metadata(similarity_df):
    """
    Extract page metadata from similarity dataframe
    Returns DataFrame with page_id and publication_name
    """
    # Get unique page info from both columns
    pages1 = similarity_df[['page1_id', 'pub1']].rename(
        columns={'page1_id': 'page_id', 'pub1': 'publication_name'}
    ).drop_duplicates()
    
    pages2 = similarity_df[['page2_id', 'pub2']].rename(
        columns={'page2_id': 'page_id', 'pub2': 'publication_name'}
    ).drop_duplicates()
    
    page_metadata = pd.concat([pages1, pages2]).drop_duplicates('page_id')
    page_metadata = page_metadata.sort_values('page_id').reset_index(drop=True)
    
    print(f"\nPage metadata: {len(page_metadata)} pages from {page_metadata['publication_name'].nunique()} publications")
    
    return page_metadata

def create_page_level_heatmap(similarity_matrix, all_pages, page_metadata, 
                              output_file='page_similarity_heatmap.png',
                              max_pages=100):
    """
    Create page-level similarity heatmap
    If too many pages, show top N most connected pages
    """
    print(f"\nCreating page-level heatmap...")
    
    n_pages = len(all_pages)
    
    if n_pages > max_pages:
        print(f"Too many pages ({n_pages}). Showing top {max_pages} most connected pages...")
        
        # Calculate average similarity for each page (excluding self)
        avg_similarity = np.mean(similarity_matrix, axis=1) - (1.0 / n_pages)  # Subtract diagonal contribution
        
        # Get top N pages
        top_indices = np.argsort(avg_similarity)[-max_pages:][::-1]
        
        # Extract submatrix
        plot_matrix = similarity_matrix[np.ix_(top_indices, top_indices)]
        plot_pages = [all_pages[i] for i in top_indices]
        
        # Get issue info for coloring
        page_issues = []
        for page_id in plot_pages:
            page_row = page_metadata[page_metadata['page_id'] == page_id]
            if not page_row.empty and 'issue_date' in page_row.columns:
                issue = page_row['issue_date'].iloc[0]
            else:
                issue = 'Unknown'
            page_issues.append(str(issue))
    else:
        plot_matrix = similarity_matrix
        plot_pages = all_pages
        page_issues = []
        for page_id in plot_pages:
            page_row = page_metadata[page_metadata['page_id'] == page_id]
            if not page_row.empty and 'issue_date' in page_row.columns:
                issue = page_row['issue_date'].iloc[0]
            else:
                issue = 'Unknown'
            page_issues.append(str(issue))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create heatmap
    sns.heatmap(
        plot_matrix,
        cmap='YlOrRd',
        vmin=0.3,
        vmax=0.9,
        cbar_kws={'label': 'Semantic Similarity'},
        square=True,
        linewidths=0,
        ax=ax,
        xticklabels=False,
        yticklabels=False
    )
    
    title = f'Page-Level Semantic Similarity Heatmap'
    if n_pages > max_pages:
        title += f'\n(Top {max_pages} most connected pages of {n_pages} total)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add issue color bar on the side
    unique_issues = sorted(set(page_issues))
    issue_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_issues)))
    issue_to_color = {issue: issue_colors[i] for i, issue in enumerate(unique_issues)}
    
    # Create color bars for issues
    colors = [issue_to_color[issue] for issue in page_issues]
    
    # Add colored bars on left and top
    ax_left = fig.add_axes([0.08, 0.125, 0.01, 0.755])
    ax_left.imshow(np.array(colors).reshape(-1, 1), aspect='auto', interpolation='nearest')
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    
    ax_top = fig.add_axes([0.125, 0.89, 0.755, 0.01])
    ax_top.imshow(np.array(colors).reshape(1, -1), aspect='auto', interpolation='nearest')
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    
    # Add legend for issues (limit to avoid overcrowding)
    from matplotlib.patches import Patch
    if len(unique_issues) <= 15:  # Show all if reasonable number
        legend_elements = [Patch(facecolor=issue_to_color[issue], label=issue) 
                          for issue in unique_issues]
        fig.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(0.01, 0.99), fontsize=8)
    else:  # Show sample if too many
        sample_issues = unique_issues[::len(unique_issues)//10]  # Sample every nth issue
        legend_elements = [Patch(facecolor=issue_to_color[issue], label=issue) 
                          for issue in sample_issues]
        fig.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(0.01, 0.99), fontsize=8, 
                  title=f"Issues (showing {len(sample_issues)} of {len(unique_issues)})")
    
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved page-level heatmap to: {OUTPUT_DIR / output_file}")
    plt.close()

def create_clustered_page_heatmap(similarity_matrix, all_pages, page_metadata,
                                  output_file='page_clustered_heatmap.png',
                                  max_pages=100):
    """
    Create hierarchically clustered page-level heatmap
    """
    print(f"\nCreating clustered page-level heatmap...")
    
    n_pages = len(all_pages)
    
    if n_pages > max_pages:
        print(f"Too many pages ({n_pages}). Showing top {max_pages} most connected pages...")
        avg_similarity = np.mean(similarity_matrix, axis=1) - (1.0 / n_pages)
        top_indices = np.argsort(avg_similarity)[-max_pages:][::-1]
        plot_matrix = similarity_matrix[np.ix_(top_indices, top_indices)]
        plot_pages = [all_pages[i] for i in top_indices]
    else:
        plot_matrix = similarity_matrix
        plot_pages = all_pages
    
    # Convert similarity to distance
    distance_matrix = 1 - plot_matrix
    distance_matrix = np.maximum(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage = hierarchy.linkage(condensed_dist, method='average')
    
    # Create clustered heatmap
    g = sns.clustermap(
        plot_matrix,
        row_linkage=linkage,
        col_linkage=linkage,
        cmap='YlOrRd',
        vmin=0.3,
        vmax=0.9,
        figsize=(16, 14),
        cbar_kws={'label': 'Semantic Similarity'},
        xticklabels=False,
        yticklabels=False,
        linewidths=0
    )
    
    title = f'Hierarchically Clustered Page-Level Similarity'
    if n_pages > max_pages:
        title += f'\n(Top {max_pages} of {n_pages} pages, grouped by similarity)'
    g.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved clustered heatmap to: {OUTPUT_DIR / output_file}")
    plt.close()

def create_page_network(similarity_df, page_metadata,
                       output_file='page_network_clusters.png',
                       threshold=0.6,
                       max_nodes=50):
    """
    Create network visualization at page level
    Show only strong connections (above threshold)
    """
    print(f"\nCreating page-level network (threshold={threshold})...")
    
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Skipping network visualization.")
        return
    
    # Filter to high similarity pairs
    high_sim = similarity_df[similarity_df['similarity'] >= threshold].copy()
    
    print(f"  Page pairs above threshold: {len(high_sim)}")
    
    if len(high_sim) == 0:
        print(f"  No pairs above threshold {threshold}. Lowering to 0.5...")
        threshold = 0.5
        high_sim = similarity_df[similarity_df['similarity'] >= threshold].copy()
    
    if len(high_sim) == 0:
        print("  Still no pairs. Cannot create network.")
        return
    
    # Create graph
    G = nx.Graph()
    
    # Add edges
    for _, row in high_sim.iterrows():
        G.add_edge(row['page1_id'], row['page2_id'], 
                  weight=row['similarity'],
                  pub1=row['pub1'],
                  pub2=row['pub2'])
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # If too many nodes, keep only most connected
    if G.number_of_nodes() > max_nodes:
        print(f"  Too many nodes. Keeping top {max_nodes} most connected...")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_ids).copy()
        print(f"  Filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Get publication for each node
    node_pubs = {}
    for node in G.nodes():
        pub = page_metadata[page_metadata['page_id'] == node]['publication_name'].iloc[0]
        node_pubs[node] = pub
    
    # Color by publication
    publications = sorted(page_metadata['publication_name'].unique())
    pub_colors = plt.cm.tab10(np.linspace(0, 1, len(publications)))
    pub_to_color = {pub: pub_colors[i] for i, pub in enumerate(publications)}
    
    node_colors = [pub_to_color[node_pubs[node]] for node in G.nodes()]
    
    # Node sizes based on degree
    node_sizes = [G.degree(node) * 100 + 50 for node in G.nodes()]
    
    # Edge widths based on similarity
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        edge_widths = [3 * (w - min_weight) / (max_weight - min_weight + 0.01) + 0.5 
                      for w in edge_weights]
    else:
        edge_widths = [1] * len(G.edges())
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=1,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.4,
        edge_color='gray',
        ax=ax
    )
    
    # Only label highly connected nodes
    high_degree_nodes = {node: f"P{node}" for node in G.nodes() if G.degree(node) > 3}
    nx.draw_networkx_labels(
        G, pos,
        labels=high_degree_nodes,
        font_size=7,
        font_weight='bold',
        ax=ax
    )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=pub_to_color[pub], label=pub) 
                      for pub in publications]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             title='Publications')
    
    title = f'Page-Level Network Based on Semantic Similarity\n(Connections show similarity ≥ {threshold:.2f}'
    if G.number_of_nodes() < len(page_metadata):
        title += f', showing top {G.number_of_nodes()} most connected pages)'
    else:
        title += ')'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    ax.margins(0.1)
    
    # Add info text
    info_text = f"Node size = number of connections\nEdge width = similarity strength\nNode color = publication"
    ax.text(0.02, 0.02, info_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved page network to: {OUTPUT_DIR / output_file}")
    plt.close()

def create_within_vs_cross_comparison(similarity_df, output_file='within_vs_cross_comparison.png'):
    """
    Compare within-publication vs cross-publication similarity distributions
    """
    print("\nCreating within vs cross-publication comparison...")
    
    # Separate within and cross publication similarities
    within = similarity_df[similarity_df['same_publication'] == True]['similarity']
    cross = similarity_df[similarity_df['same_publication'] == False]['similarity']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution plots
    ax1.hist(within, bins=30, alpha=0.7, label='Within-publication', color='steelblue', edgecolor='black')
    ax1.hist(cross, bins=30, alpha=0.7, label='Cross-publication', color='coral', edgecolor='black')
    ax1.set_xlabel('Semantic Similarity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Page-Level Similarity Scores', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Box plots
    data_to_plot = [within, cross]
    bp = ax2.boxplot(data_to_plot, labels=['Within-publication', 'Cross-publication'],
                     patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Semantic Similarity', fontsize=12)
    ax2.set_title('Page-Level Similarity Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Within-pub mean: {within.mean():.3f}\nCross-pub mean: {cross.mean():.3f}\nDifference: {within.mean() - cross.mean():.3f}"
    ax2.text(0.98, 0.98, stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {OUTPUT_DIR / output_file}")
    plt.close()

def generate_summary_statistics(similarity_df, page_metadata):
    """
    Generate and print summary statistics
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (PAGE LEVEL)")
    print("="*60)
    
    # Overall statistics
    print("\nOverall:")
    print(f"  Total pages: {len(page_metadata)}")
    print(f"  Total page pairs analyzed: {len(similarity_df):,}")
    print(f"  Average similarity: {similarity_df['similarity'].mean():.3f}")
    print(f"  Similarity range: [{similarity_df['similarity'].min():.3f}, {similarity_df['similarity'].max():.3f}]")
    
    # Within vs cross publication
    within = similarity_df[similarity_df['same_publication'] == True]
    cross = similarity_df[similarity_df['same_publication'] == False]
    
    print("\nWithin-publication (pages from same pub):")
    print(f"  Pairs: {len(within):,}")
    print(f"  Average similarity: {within['similarity'].mean():.3f}")
    print(f"  Std dev: {within['similarity'].std():.3f}")
    
    print("\nCross-publication (pages from different pubs):")
    print(f"  Pairs: {len(cross):,}")
    print(f"  Average similarity: {cross['similarity'].mean():.3f}")
    print(f"  Std dev: {cross['similarity'].std():.3f}")
    
    print(f"\nDifference (within - cross): {within['similarity'].mean() - cross['similarity'].mean():.3f}")
    
    # Per-publication statistics
    print("\n" + "="*60)
    print("PER-PUBLICATION STATISTICS")
    print("="*60)
    
    for pub in sorted(page_metadata['publication_name'].unique()):
        pub_pages = page_metadata[page_metadata['publication_name'] == pub]['page_id']
        n_pages = len(pub_pages)
        
        # Get within-pub similarities for this publication
        pub_within = within[within['pub1'] == pub]
        
        if len(pub_within) > 0:
            avg_sim = pub_within['similarity'].mean()
            print(f"\n{pub}:")
            print(f"  Pages: {n_pages}")
            print(f"  Avg internal similarity: {avg_sim:.3f}")
            print(f"  Internal comparisons: {len(pub_within)}")
        else:
            print(f"\n{pub}:")
            print(f"  Pages: {n_pages}")
            print(f"  (Only 1 page, no internal comparisons)")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("SEMANTIC SIMILARITY VISUALIZATION (PAGE LEVEL)")
    print("Working from all_page_similarities.csv")
    print("="*60)
    
    # 1. Load and filter data
    print("\nSTEP 1: Loading and filtering similarity data")
    print("-" * 60)
    similarity_df = load_filtered_similarity_data()
    
    if len(similarity_df) == 0:
        print("ERROR: No data remaining after filtering!")
        return
    
    # 2. Get page metadata
    print("\nSTEP 2: Extracting page metadata")
    print("-" * 60)
    page_metadata = get_page_metadata(similarity_df)
    
    # 3. Create full similarity matrix
    print("\nSTEP 3: Creating similarity matrix")
    print("-" * 60)
    similarity_matrix, all_pages = create_similarity_matrix_from_pairs(similarity_df)
    
    # 4. Create visualizations
    print("\nSTEP 4: Creating visualizations")
    print("-" * 60)
    
    # Page-level heatmap
    create_page_level_heatmap(similarity_matrix, all_pages, page_metadata, max_pages=100)
    
    # Clustered heatmap
    create_clustered_page_heatmap(similarity_matrix, all_pages, page_metadata, max_pages=100)
    
    # Network clustering
    create_page_network(similarity_df, page_metadata, threshold=0.6, max_nodes=50)
    
    # Within vs cross comparison
    create_within_vs_cross_comparison(similarity_df)
    
    # 5. Generate summary statistics
    print("\nSTEP 5: Generating summary statistics")
    print("-" * 60)
    generate_summary_statistics(similarity_df, page_metadata)
    
    # Final summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - page_similarities_no_citations.csv      (Filtered data)")
    print("  - page_similarity_heatmap.png             (Page-level heatmap)")
    print("  - page_clustered_heatmap.png              (Hierarchical clustering)")
    print("  - page_network_clusters.png               (Network visualization)")
    print("  - within_vs_cross_comparison.png          (Distribution comparison)")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()