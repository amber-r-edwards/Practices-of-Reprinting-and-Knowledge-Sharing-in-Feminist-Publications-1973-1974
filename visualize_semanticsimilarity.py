"""
Semantic Similarity Visualization Script
Works from all_page_similarities.csv with citation filtering
Generates publication-level heatmap and clustering visualizations
"""

import pandas as pd # type: ignore # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from pathlib import Path
from scipy.cluster import hierarchy # type: ignore
from scipy.spatial.distance import squareform # type: ignore

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

def aggregate_to_publication_level(similarity_df):
    """
    Aggregate page-level similarities to publication-level
    Returns publication × publication similarity matrix
    """
    print("\nAggregating to publication level...")
    
    publications = sorted(pd.concat([
        similarity_df['pub1'], 
        similarity_df['pub2']
    ]).unique())
    
    n_pubs = len(publications)
    print(f"Publications: {n_pubs}")
    
    # Initialize publication similarity matrix
    pub_similarity = np.zeros((n_pubs, n_pubs))
    pub_counts = np.zeros((n_pubs, n_pubs))
    
    # Create publication index mapping
    pub_to_idx = {pub: i for i, pub in enumerate(publications)}
    
    # Aggregate similarities
    for _, row in similarity_df.iterrows():
        i = pub_to_idx[row['pub1']]
        j = pub_to_idx[row['pub2']]
        
        # Add to both directions (symmetric matrix)
        pub_similarity[i, j] += row['similarity']
        pub_similarity[j, i] += row['similarity']
        pub_counts[i, j] += 1
        pub_counts[j, i] += 1
    
    # Average the similarities
    with np.errstate(divide='ignore', invalid='ignore'):
        pub_similarity = np.where(pub_counts > 0, pub_similarity / pub_counts, 0)
    
    # Handle within-publication (diagonal)
    # Get within-publication similarities
    within_pub_sim = similarity_df[similarity_df['same_publication'] == True]
    for pub in publications:
        pub_data = within_pub_sim[within_pub_sim['pub1'] == pub]
        if len(pub_data) > 0:
            idx = pub_to_idx[pub]
            pub_similarity[idx, idx] = pub_data['similarity'].mean()
    
    # Create DataFrame
    pub_sim_df = pd.DataFrame(
        pub_similarity,
        index=publications,
        columns=publications
    )
    
    # Print statistics
    print("\nPublication-level statistics:")
    for pub in publications:
        n_pages = similarity_df[
            (similarity_df['pub1'] == pub) | (similarity_df['pub2'] == pub)
        ]['page1_id'].nunique()
        print(f"  {pub}: avg similarity = {pub_sim_df.loc[pub, pub]:.3f}")
    
    return pub_sim_df

def create_publication_heatmap(pub_sim_df, output_file='publication_similarity_heatmap.png'):
    """
    Create publication-level similarity heatmap
    """
    print("\nCreating publication similarity heatmap...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        pub_sim_df,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0.4,  # Adjust based on your data
        vmax=0.8,
        cbar_kws={'label': 'Average Semantic Similarity'},
        linewidths=0.5,
        square=True,
        ax=ax
    )
    
    ax.set_title('Semantic Similarity Between Publications\n(Higher = More Ideologically Aligned)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Publication', fontsize=13)
    ax.set_ylabel('Publication', fontsize=13)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {OUTPUT_DIR / output_file}")
    plt.close()

def create_clustered_heatmap(pub_sim_df, output_file='publication_clustered_heatmap.png'):
    """
    Create hierarchically clustered heatmap showing publication relationships
    """
    print("\nCreating clustered heatmap...")
    
    # Convert similarity to distance for clustering
    # Distance = 1 - similarity
    distance_matrix = 1 - pub_sim_df.values
    
    # Ensure symmetric and non-negative
    distance_matrix = np.maximum(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage = hierarchy.linkage(condensed_dist, method='average')
    
    # Create figure with dendrogram
    fig = plt.figure(figsize=(14, 12))
    
    # Create gridspec for layout
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    
    # Top dendrogram
    ax_top = fig.add_subplot(gs[0, 1])
    dendro_top = hierarchy.dendrogram(
        linkage,
        ax=ax_top,
        orientation='top',
        labels=pub_sim_df.columns,
        no_labels=True,
        color_threshold=0,
        above_threshold_color='gray'
    )
    ax_top.axis('off')
    
    # Left dendrogram
    ax_left = fig.add_subplot(gs[1, 0])
    dendro_left = hierarchy.dendrogram(
        linkage,
        ax=ax_left,
        orientation='left',
        labels=pub_sim_df.index,
        no_labels=True,
        color_threshold=0,
        above_threshold_color='gray'
    )
    ax_left.axis('off')
    
    # Reorder matrix based on clustering
    order = dendro_top['leaves']
    ordered_matrix = pub_sim_df.iloc[order, order]
    
    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        ordered_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0.4,
        vmax=0.8,
        cbar_kws={'label': 'Semantic Similarity'},
        linewidths=0.5,
        square=True,
        ax=ax_heatmap
    )
    
    ax_heatmap.set_xlabel('Publication', fontsize=12)
    ax_heatmap.set_ylabel('Publication', fontsize=12)
    plt.setp(ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Overall title
    fig.suptitle('Hierarchically Clustered Publication Similarity\n(Publications grouped by ideological alignment)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved clustered heatmap to: {OUTPUT_DIR / output_file}")
    plt.close()

def create_network_clustering(pub_sim_df, output_file='publication_network_clusters.png',
                              threshold=0.5):
    """
    Create network visualization with publications as nodes
    Show only strong connections (above threshold)
    """
    print(f"\nCreating network clustering (threshold={threshold})...")
    
    try:
        import networkx as nx # type: ignore
    except ImportError:
        print("NetworkX not installed. Skipping network visualization.")
        return
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    publications = pub_sim_df.index.tolist()
    G.add_nodes_from(publications)
    
    # Add edges for similarities above threshold
    edges_added = 0
    for i, pub1 in enumerate(publications):
        for j, pub2 in enumerate(publications):
            if i < j:  # Avoid duplicates
                similarity = pub_sim_df.iloc[i, j]
                if similarity >= threshold:
                    G.add_edge(pub1, pub2, weight=similarity)
                    edges_added += 1
    
    print(f"  Added {edges_added} edges above threshold {threshold}")
    
    if edges_added == 0:
        print(f"  No edges above threshold {threshold}. Lowering threshold to 0.4...")
        threshold = 0.4
        for i, pub1 in enumerate(publications):
            for j, pub2 in enumerate(publications):
                if i < j:
                    similarity = pub_sim_df.iloc[i, j]
                    if similarity >= threshold:
                        G.add_edge(pub1, pub2, weight=similarity)
                        edges_added += 1
    
    if edges_added == 0:
        print("  Still no edges. All publications may be very dissimilar.")
        # Add all edges for visualization
        for i, pub1 in enumerate(publications):
            for j, pub2 in enumerate(publications):
                if i < j:
                    similarity = pub_sim_df.iloc[i, j]
                    G.add_edge(pub1, pub2, weight=similarity)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on average similarity to all others
    node_sizes = []
    for pub in publications:
        avg_sim = pub_sim_df.loc[pub].mean()
        node_sizes.append(avg_sim * 5000)
    
    # Edge widths based on similarity
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        edge_widths = [5 * (w - min_weight) / (max_weight - min_weight + 0.01) + 1 
                      for w in edge_weights]
    else:
        edge_widths = [2] * len(G.edges())
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.5,
        edge_color='gray',
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_weight='bold',
        font_color='black',
        ax=ax
    )
    
    ax.set_title(f'Publication Network Based on Semantic Similarity\n(Connections show similarity ≥ {threshold:.2f})',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    ax.margins(0.15)
    
    # Add legend
    legend_text = f"Node size = avg similarity to all publications\nEdge width = pairwise similarity\nShowing edges ≥ {threshold:.2f}"
    ax.text(0.02, 0.98, legend_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved network clustering to: {OUTPUT_DIR / output_file}")
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
    ax1.set_title('Distribution of Similarity Scores', fontsize=14, fontweight='bold')
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
    ax2.set_title('Similarity Score Comparison', fontsize=14, fontweight='bold')
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

def generate_summary_statistics(similarity_df, pub_sim_df):
    """
    Generate and print summary statistics
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print("\nOverall:")
    print(f"  Total page pairs analyzed: {len(similarity_df):,}")
    print(f"  Average similarity: {similarity_df['similarity'].mean():.3f}")
    print(f"  Similarity range: [{similarity_df['similarity'].min():.3f}, {similarity_df['similarity'].max():.3f}]")
    
    # Within vs cross publication
    within = similarity_df[similarity_df['same_publication'] == True]
    cross = similarity_df[similarity_df['same_publication'] == False]
    
    print("\nWithin-publication:")
    print(f"  Pairs: {len(within):,}")
    print(f"  Average similarity: {within['similarity'].mean():.3f}")
    print(f"  Std dev: {within['similarity'].std():.3f}")
    
    print("\nCross-publication:")
    print(f"  Pairs: {len(cross):,}")
    print(f"  Average similarity: {cross['similarity'].mean():.3f}")
    print(f"  Std dev: {cross['similarity'].std():.3f}")
    
    print(f"\nDifference (within - cross): {within['similarity'].mean() - cross['similarity'].mean():.3f}")
    
    # Publication-level statistics
    print("\n" + "="*60)
    print("PUBLICATION-LEVEL STATISTICS")
    print("="*60)
    
    print("\nInternal coherence (diagonal values):")
    for pub in pub_sim_df.index:
        print(f"  {pub}: {pub_sim_df.loc[pub, pub]:.3f}")
    
    print("\nTop 5 cross-publication similarities:")
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(pub_sim_df), k=1).astype(bool)
    upper_tri = pub_sim_df.where(mask)
    stacked = upper_tri.stack().sort_values(ascending=False)
    for (pub1, pub2), sim in stacked.head(5).items():
        print(f"  {pub1} ↔ {pub2}: {sim:.3f}")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("SEMANTIC SIMILARITY VISUALIZATION")
    print("Working from all_page_similarities.csv")
    print("="*60)
    
    # 1. Load and filter data
    print("\nSTEP 1: Loading and filtering similarity data")
    print("-" * 60)
    similarity_df = load_filtered_similarity_data()
    
    if len(similarity_df) == 0:
        print("ERROR: No data remaining after filtering!")
        return
    
    # 2. Aggregate to publication level
    print("\nSTEP 2: Aggregating to publication level")
    print("-" * 60)
    pub_sim_df = aggregate_to_publication_level(similarity_df)
    
    # 3. Create visualizations
    print("\nSTEP 3: Creating visualizations")
    print("-" * 60)
    
    # Basic heatmap
    create_publication_heatmap(pub_sim_df)
    
    # Clustered heatmap
    create_clustered_heatmap(pub_sim_df)
    
    # Network clustering
    create_network_clustering(pub_sim_df, threshold=0.5)
    
    # Within vs cross comparison
    create_within_vs_cross_comparison(similarity_df)
    
    # 4. Generate summary statistics
    print("\nSTEP 4: Generating summary statistics")
    print("-" * 60)
    generate_summary_statistics(similarity_df, pub_sim_df)
    
    # Final summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - page_similarities_no_citations.csv      (Filtered data)")
    print("  - publication_similarity_heatmap.png      (Basic heatmap)")
    print("  - publication_clustered_heatmap.png       (Hierarchical clustering)")
    print("  - publication_network_clusters.png        (Network visualization)")
    print("  - within_vs_cross_comparison.png          (Distribution comparison)")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()