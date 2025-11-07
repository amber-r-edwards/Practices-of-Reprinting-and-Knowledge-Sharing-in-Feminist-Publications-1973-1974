"""
Semantic Similarity Analysis for Feminist Publications
Analyzes ideological alignment across publications using sentence embeddings
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path('semantic_results')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_metadata(filepath='page_metadata.csv'):
    """Load page metadata"""
    metadata = pd.read_csv(filepath)
    
    # Ensure we have the required columns
    required_cols = ['page_id', 'publication_name', 'text']
    missing = [col for col in required_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add page_id if not present
    if 'page_id' not in metadata.columns:
        metadata['page_id'] = range(len(metadata))
    
    # Clean text
    metadata['text_clean'] = metadata['text'].fillna('').astype(str)
    
    # Filter out empty pages
    metadata = metadata[metadata['text_clean'].str.len() > 50].copy()
    
    print(f"Loaded {len(metadata)} pages from {metadata['publication_name'].nunique()} publications")
    return metadata

def generate_page_embeddings(metadata, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Generate embeddings at page level, store with page_id
    
    Parameters:
    - metadata: DataFrame with page_id and text_clean
    - model_name: 'all-MiniLM-L6-v2' (384 dimensions)
    - batch_size: number of pages to process at once
    
    Returns:
    - DataFrame with page_id and embeddings
    """
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(metadata)} pages...")
    embeddings = model.encode(
        metadata['text_clean'].tolist(), 
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Embeddings shape: {embeddings.shape} (pages × dimensions)")
    
    # Create DataFrame with page_id and embeddings
    embeddings_df = pd.DataFrame({
        'page_id': metadata['page_id'].values,
        'embedding': list(embeddings)  # Store as list of arrays
    })
    
    # Merge with metadata to keep publication info
    embeddings_df = embeddings_df.merge(
        metadata[['page_id', 'publication_name', 'issue_date']],
        on='page_id',
        how='left'
    )
    
    return embeddings_df, embeddings

def calculate_similarity_matrix(embeddings):
    """Calculate cosine similarity between all embeddings"""
    print("\nCalculating similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix

def create_publication_similarity_matrix(similarity_matrix, metadata):
    """
    Aggregate page-level similarities to publication-level
    """
    publications = metadata['publication_name'].unique()
    n_pubs = len(publications)
    
    # Create publication similarity matrix
    pub_similarity = np.zeros((n_pubs, n_pubs))
    
    # Debug: Check dimensions
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Number of publications: {n_pubs}")
    
    for i, pub1 in enumerate(publications):
        for j, pub2 in enumerate(publications):
            # Get page indices for each publication
            pages1 = metadata[metadata['publication_name'] == pub1].index.tolist()
            pages2 = metadata[metadata['publication_name'] == pub2].index.tolist()
            
            # Debug: Check if indices are within bounds
            max_idx1 = max(pages1) if pages1 else -1
            max_idx2 = max(pages2) if pages2 else -1
            
            if max_idx1 >= similarity_matrix.shape[0] or max_idx2 >= similarity_matrix.shape[1]:
                print(f"WARNING: Index out of bounds for {pub1} vs {pub2}")
                print(f"  Max index pub1: {max_idx1}, pub2: {max_idx2}")
                print(f"  Matrix shape: {similarity_matrix.shape}")
                continue
            
            if len(pages1) > 0 and len(pages2) > 0:
                # Extract submatrix for these publications
                try:
                    submatrix = similarity_matrix[np.ix_(pages1, pages2)]
                    # Average similarity between publications
                    pub_similarity[i, j] = np.mean(submatrix)
                except IndexError as e:
                    print(f"Error with {pub1} vs {pub2}: {e}")
                    print(f"  pages1: {pages1[:5]}... (showing first 5)")
                    print(f"  pages2: {pages2[:5]}... (showing first 5)")
                    pub_similarity[i, j] = 0
    
    # Convert to DataFrame
    pub_sim_df = pd.DataFrame(
        pub_similarity,
        index=publications,
        columns=publications
    )
    
    return pub_sim_df

def visualize_publication_heatmap(pub_sim_df, output_file='publication_similarity_heatmap.png'):
    """
    Create heatmap of publication-level semantic similarity
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        pub_sim_df,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0.5,  # Adjust based on your data
        vmax=1.0,
        cbar_kws={'label': 'Average Semantic Similarity'},
        linewidths=0.5,
        square=True,
        ax=ax
    )
    
    ax.set_title('Semantic Similarity Between Publications\n(Higher = More Ideologically Aligned)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Publication', fontsize=12)
    ax.set_ylabel('Publication', fontsize=12)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"Saved publication heatmap to {OUTPUT_DIR / output_file}")
    plt.close()

def analyze_within_vs_cross_publication(pub_sim_df):
    """
    Compare within-publication vs cross-publication similarity
    """
    publications = pub_sim_df.index.tolist()
    
    # Within-publication similarities (diagonal)
    within_pub = {}
    for i, pub in enumerate(publications):
        within_pub[pub] = pub_sim_df.iloc[i, i]
    
    # Cross-publication similarities (off-diagonal)
    cross_pub = []
    for i, pub1 in enumerate(publications):
        for j, pub2 in enumerate(publications):
            if i != j:
                cross_pub.append({
                    'pub1': pub1,
                    'pub2': pub2,
                    'similarity': pub_sim_df.iloc[i, j]
                })
    
    cross_pub_df = pd.DataFrame(cross_pub)
    
    print("\n" + "="*60)
    print("WITHIN-PUBLICATION COHERENCE")
    print("="*60)
    print("(How semantically consistent each publication is internally)\n")
    for pub, sim in sorted(within_pub.items(), key=lambda x: x[1], reverse=True):
        print(f"{pub}: {sim:.3f}")
    
    print("\n" + "="*60)
    print("TOP CROSS-PUBLICATION SIMILARITIES")
    print("="*60)
    print("(Strongest ideological alignment between publications)\n")
    top_cross = cross_pub_df.nlargest(10, 'similarity')
    for _, row in top_cross.iterrows():
        print(f"{row['pub1']} ↔ {row['pub2']}: {row['similarity']:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Average within-publication similarity: {np.mean(list(within_pub.values())):.3f}")
    print(f"Average cross-publication similarity: {cross_pub_df['similarity'].mean():.3f}")
    print(f"Difference: {np.mean(list(within_pub.values())) - cross_pub_df['similarity'].mean():.3f}")
    
    return within_pub, cross_pub_df

def create_temporal_similarity(similarity_matrix, metadata, output_file='temporal_similarity.png'):
    """
    Analyze how similarity changes over time
    """
    if 'issue_date' not in metadata.columns:
        print("No date information available for temporal analysis")
        return
    
    # Parse dates
    metadata['issue_date'] = pd.to_datetime(metadata['issue_date'], errors='coerce')
    metadata = metadata.dropna(subset=['issue_date'])
    
    if len(metadata) == 0:
        print("No valid dates for temporal analysis")
        return
    
    # Add month column
    metadata['month'] = metadata['issue_date'].dt.to_period('M')
    months = sorted(metadata['month'].unique())
    
    # Calculate average similarity by month pair
    temporal_sim = np.zeros((len(months), len(months)))
    
    for i, month1 in enumerate(months):
        for j, month2 in enumerate(months):
            pages1 = metadata[metadata['month'] == month1].index.tolist()
            pages2 = metadata[metadata['month'] == month2].index.tolist()
            
            if pages1 and pages2:
                submatrix = similarity_matrix[np.ix_(pages1, pages2)]
                temporal_sim[i][j] = submatrix.mean()
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(temporal_sim, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    
    ax.set_xticks(range(len(months)))
    ax.set_yticks(range(len(months)))
    ax.set_xticklabels([str(m) for m in months], rotation=45, ha='right')
    ax.set_yticklabels([str(m) for m in months])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Semantic Similarity', rotation=270, labelpad=20)
    
    ax.set_title('Semantic Similarity Over Time\n(How ideologically aligned different time periods are)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_file, dpi=300, bbox_inches='tight')
    print(f"Saved temporal similarity to {OUTPUT_DIR / output_file}")
    plt.close()

def find_most_similar_pages(similarity_matrix, metadata, n_top=20):
    """
    Find the most semantically similar page pairs
    """
    print(f"\nFinding top {n_top} most similar page pairs...")
    
    # Get upper triangle indices (avoid duplicates and self-comparisons)
    rows, cols = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[rows, cols]
    
    # Get top N
    top_indices = np.argsort(similarities)[-n_top:][::-1]
    
    results = []
    for idx in top_indices:
        i, j = rows[idx], cols[idx]
        
        page1 = metadata.iloc[i]
        page2 = metadata.iloc[j]
        
        results.append({
            'page1_id': page1['page_id'],
            'page2_id': page2['page_id'],
            'pub1': page1['publication_name'],
            'pub2': page2['publication_name'],
            'similarity': similarities[idx],
            'same_publication': page1['publication_name'] == page2['publication_name'],
            'text1_preview': page1['text_clean'][:100],
            'text2_preview': page2['text_clean'][:100]
        })
    
    similar_pages_df = pd.DataFrame(results)
    similar_pages_df.to_csv(OUTPUT_DIR / 'most_similar_pages.csv', index=False)
    
    print("\n" + "="*60)
    print("TOP SEMANTICALLY SIMILAR PAGE PAIRS")
    print("="*60)
    for _, row in similar_pages_df.head(10).iterrows():
        same = "SAME PUB" if row['same_publication'] else "CROSS-PUB"
        print(f"\n{row['pub1']} (page {row['page1_id']}) ↔ {row['pub2']} (page {row['page2_id']})")
        print(f"Similarity: {row['similarity']:.3f} [{same}]")
    
    return similar_pages_df

def save_similarity_data(similarity_matrix, metadata, pub_sim_df):
    """
    Save similarity matrices and metadata for later use
    """
    print("\nSaving similarity data...")
    
    # Save page-level similarity matrix (compressed)
    np.save(OUTPUT_DIR / 'page_similarity_matrix.npy', similarity_matrix)
    
    # Save publication-level similarity
    pub_sim_df.to_csv(OUTPUT_DIR / 'publication_similarity.csv')
    
    # Save metadata with indices
    metadata.to_csv(OUTPUT_DIR / 'metadata_with_indices.csv', index=False)
    
    # Save summary statistics
    summary = {
        'n_pages': len(metadata),
        'n_publications': metadata['publication_name'].nunique(),
        'publications': metadata['publication_name'].unique().tolist(),
        'avg_within_pub_similarity': float(np.diag(pub_sim_df).mean()),
        'avg_cross_pub_similarity': float(pub_sim_df.values[~np.eye(len(pub_sim_df), dtype=bool)].mean())
    }
    
    with open(OUTPUT_DIR / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved all data to {OUTPUT_DIR}/")

def create_comparison_with_text_reuse(pub_sim_df):
    """
    Compare semantic similarity with text reuse results if available
    """
    reuse_file = Path('reuse_results/text_reuse_results.csv')
    
    if not reuse_file.exists():
        print("\nText reuse results not found. Skipping comparison.")
        return
    
    print("\nComparing with text reuse results...")
    
    # Load text reuse data
    reuse_df = pd.read_csv(reuse_file)
    
    # Aggregate text reuse by publication pair
    reuse_counts = reuse_df.groupby(['source_publication', 'target_publication']).size().reset_index(name='reuse_count')
    
    # Create comparison data
    comparison = []
    for _, row in reuse_counts.iterrows():
        src, tgt = row['source_publication'], row['target_publication']
        
        # Get semantic similarity
        if src in pub_sim_df.index and tgt in pub_sim_df.columns:
            sem_sim = pub_sim_df.loc[src, tgt]
            comparison.append({
                'source': src,
                'target': tgt,
                'text_reuse_count': row['reuse_count'],
                'semantic_similarity': sem_sim
            })
    
    if not comparison:
        print("No overlapping publication pairs found")
        return
    
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(OUTPUT_DIR / 'reuse_vs_similarity_comparison.csv', index=False)
    
    # Visualize comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        comp_df['semantic_similarity'],
        comp_df['text_reuse_count'],
        s=100,
        alpha=0.6,
        c=comp_df['semantic_similarity'],
        cmap='viridis'
    )
    
    # Add labels for each point
    for _, row in comp_df.iterrows():
        ax.annotate(
            f"{row['source'][:10]}->{row['target'][:10]}",
            (row['semantic_similarity'], row['text_reuse_count']),
            fontsize=8,
            alpha=0.7
        )
    
    ax.set_xlabel('Semantic Similarity (Ideological Alignment)', fontsize=12)
    ax.set_ylabel('Text Reuse Count (Direct Sharing)', fontsize=12)
    ax.set_title('Semantic Similarity vs. Text Reuse\n(Do aligned publications also share text directly?)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, label='Semantic Similarity')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'reuse_vs_similarity_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {OUTPUT_DIR}/reuse_vs_similarity_scatter.png")
    plt.close()
    
    # Print correlation
    if len(comp_df) > 2:
        corr = comp_df['semantic_similarity'].corr(comp_df['text_reuse_count'])
        print(f"\nCorrelation between semantic similarity and text reuse: {corr:.3f}")
        print("(Positive = publications that are ideologically aligned also share text)")
        print("(Near zero = ideological alignment independent of text sharing)")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("="*60)
    
    # 1. Load data
    metadata = load_metadata('page_metadata.csv')
    
    # Reset index to ensure it matches similarity matrix dimensions
    metadata = metadata.reset_index(drop=True)
    
    # 2. Generate embeddings - FIXED: pass the full metadata DataFrame
    embeddings_df, embeddings = generate_page_embeddings(metadata)
    
    # 3. Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)
    
    # 4. Create publication-level similarity
    pub_sim_df = create_publication_similarity_matrix(similarity_matrix, metadata)
    
    # 5. Visualize publication heatmap
    visualize_publication_heatmap(pub_sim_df)
    
    # 6. Analyze within vs cross-publication
    within_pub, cross_pub_df = analyze_within_vs_cross_publication(pub_sim_df)
    
    # 7. Temporal analysis (if dates available)
    create_temporal_similarity(similarity_matrix, metadata)
    
    # 8. Find most similar pages
    similar_pages = find_most_similar_pages(similarity_matrix, metadata)
    
    # 9. Save all data
    save_similarity_data(similarity_matrix, metadata, pub_sim_df)
    
    # 10. Compare with text reuse (if available)
    create_comparison_with_text_reuse(pub_sim_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - publication_similarity_heatmap.png")
    print("  - temporal_similarity.png (if dates available)")
    print("  - reuse_vs_similarity_scatter.png (if text reuse data available)")
    print("  - publication_similarity.csv")
    print("  - most_similar_pages.csv")
    print("  - page_similarity_matrix.npy")
    print("  - summary_stats.json")

if __name__ == "__main__":
    main()