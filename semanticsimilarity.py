"""
Semantic Similarity Analysis for Feminist Publications
Analyzes ideological alignment across publications using sentence embeddings
CSV outputs only - no visualizations
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json

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
    
    # Clean text
    metadata['text_clean'] = metadata['text'].fillna('').astype(str)
    
    # Filter out empty pages
    metadata = metadata[metadata['text_clean'].str.len() > 50].copy()
    
    # Reset index to ensure it matches similarity matrix dimensions
    metadata = metadata.reset_index(drop=True)
    
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
    - numpy array of embeddings
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

def calculate_page_similarity_matrix(embeddings):
    """
    Calculate cosine similarity between all page embeddings
    
    Returns:
    - similarity_matrix: (n_pages × n_pages) matrix where [i,j] = similarity between page i and page j
    """
    print("\nCalculating page-level similarity matrix...")
    print(f"Computing {len(embeddings)} × {len(embeddings)} pairwise similarities...")
    
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    return similarity_matrix

def export_all_page_similarities(similarity_matrix, metadata, output_file='all_page_similarities.csv'):
    """
    Export ALL pairwise page similarities to CSV (not just top N)
    This allows filtering out citations later
    
    Output columns:
    - page1_id, page2_id
    - pub1, pub2
    - similarity
    - same_publication
    - page1_text_preview, page2_text_preview (for manual review)
    """
    print("\nExporting all pairwise page similarities...")
    print("WARNING: This will be a large file for many pages")
    
    n_pages = len(metadata)
    print(f"Total comparisons to export: {n_pages * (n_pages - 1) // 2}")
    
    # Get upper triangle indices (avoid duplicates and self-comparisons)
    rows, cols = np.triu_indices_from(similarity_matrix, k=1)
    
    results = []
    batch_size = 10000
    
    for batch_start in range(0, len(rows), batch_size):
        batch_end = min(batch_start + batch_size, len(rows))
        
        if batch_start % 50000 == 0:
            print(f"Processing comparisons {batch_start} to {batch_end}...")
        
        for idx in range(batch_start, batch_end):
            i, j = rows[idx], cols[idx]
            
            page1 = metadata.iloc[i]
            page2 = metadata.iloc[j]
            
            results.append({
                'page1_id': page1['page_id'],
                'page2_id': page2['page_id'],
                'pub1': page1['publication_name'],
                'pub2': page2['publication_name'],
                'similarity': similarity_matrix[i, j],
                'same_publication': page1['publication_name'] == page2['publication_name'],
                'page1_text_preview': page1['text_clean'][:150],
                'page2_text_preview': page2['text_clean'][:150]
            })
    
    print(f"Creating DataFrame with {len(results)} similarity pairs...")
    all_similarities_df = pd.DataFrame(results)
    
    # Sort by similarity descending
    all_similarities_df = all_similarities_df.sort_values('similarity', ascending=False)
    
    # Save to CSV
    output_path = OUTPUT_DIR / output_file
    all_similarities_df.to_csv(output_path, index=False)
    print(f"✓ Saved all page similarities to: {output_path}")
    print(f"  Total pairs: {len(all_similarities_df)}")
    print(f"  File size: ~{output_path.stat().st_size / (1024*1024):.1f} MB")
    
    return all_similarities_df

def aggregate_to_publication_level(similarity_matrix, metadata):
    """
    Aggregate page-level similarities to publication-level
    Shows similarity between all pages across publications
    
    Returns:
    - pub_similarity_df: publication × publication DataFrame of average similarities
    - within_pub_stats: statistics about within-publication coherence
    - cross_pub_stats: statistics about cross-publication alignment
    """
    publications = sorted(metadata['publication_name'].unique())
    n_pubs = len(publications)
    
    print(f"\n{'='*60}")
    print(f"AGGREGATING TO PUBLICATION LEVEL")
    print(f"{'='*60}")
    print(f"Publications: {n_pubs}")
    print(f"Total pages: {len(metadata)}")
    
    # Initialize results
    pub_similarity = np.zeros((n_pubs, n_pubs))
    pub_counts = np.zeros((n_pubs, n_pubs))
    
    for i, pub1 in enumerate(publications):
        for j, pub2 in enumerate(publications):
            # Get page indices for each publication
            pages1 = metadata[metadata['publication_name'] == pub1].index.tolist()
            pages2 = metadata[metadata['publication_name'] == pub2].index.tolist()
            
            pub_counts[i][j] = len(pages1) * len(pages2)
            
            if i == j:
                # Within-publication: exclude diagonal (self-similarity = 1.0)
                if len(pages1) > 1:
                    submatrix = similarity_matrix[np.ix_(pages1, pages2)]
                    mask = ~np.eye(len(pages1), dtype=bool)
                    pub_similarity[i][j] = submatrix[mask].mean()
                else:
                    pub_similarity[i][j] = 0  # Only one page
            else:
                # Cross-publication: average of all page-to-page similarities
                submatrix = similarity_matrix[np.ix_(pages1, pages2)]
                pub_similarity[i][j] = submatrix.mean()
            
            print(f"  {pub1} → {pub2}: {pub_similarity[i][j]:.3f} (from {int(pub_counts[i][j])} page comparisons)")
    
    # Create DataFrame
    pub_sim_df = pd.DataFrame(
        pub_similarity,
        index=publications,
        columns=publications
    )
    
    # Calculate statistics
    within_pub_stats = {}
    for i, pub in enumerate(publications):
        n_pages = len(metadata[metadata['publication_name'] == pub])
        within_pub_stats[pub] = {
            'avg_similarity': pub_similarity[i][i],
            'n_pages': n_pages,
            'n_comparisons': int(pub_counts[i][i])
        }
    
    cross_pub_pairs = []
    for i, pub1 in enumerate(publications):
        for j, pub2 in enumerate(publications):
            if i != j:
                cross_pub_pairs.append({
                    'pub1': pub1,
                    'pub2': pub2,
                    'similarity': pub_similarity[i][j],
                    'n_comparisons': int(pub_counts[i][j])
                })
    
    cross_pub_stats = pd.DataFrame(cross_pub_pairs)
    
    return pub_sim_df, within_pub_stats, cross_pub_stats

def analyze_within_vs_cross_publication(pub_sim_df, within_pub_stats, cross_pub_stats):
    """
    Analyze and report within-publication vs cross-publication similarity patterns
    """
    print("\n" + "="*60)
    print("WITHIN-PUBLICATION COHERENCE")
    print("="*60)
    print("How semantically consistent is each publication internally?")
    print("(Higher = more focused/coherent content)")
    print()
    
    sorted_within = sorted(within_pub_stats.items(), 
                          key=lambda x: x[1]['avg_similarity'], 
                          reverse=True)
    
    for pub, stats in sorted_within:
        print(f"{pub}:")
        print(f"  Avg similarity: {stats['avg_similarity']:.3f}")
        print(f"  Pages: {stats['n_pages']}")
        print(f"  Comparisons: {stats['n_comparisons']}")
        print()
    
    print("="*60)
    print("TOP CROSS-PUBLICATION SIMILARITIES")
    print("="*60)
    print("Strongest ideological alignment between different publications")
    print("(Higher = more similar content/perspectives)")
    print()
    
    top_cross = cross_pub_stats.nlargest(10, 'similarity')
    for _, row in top_cross.iterrows():
        print(f"{row['pub1']} ↔ {row['pub2']}")
        print(f"  Similarity: {row['similarity']:.3f}")
        print(f"  Comparisons: {row['n_comparisons']} page pairs")
        print()
    
    print("="*60)
    print("BOTTOM CROSS-PUBLICATION SIMILARITIES")
    print("="*60)
    print("Most distinct/divergent publications")
    print()
    
    bottom_cross = cross_pub_stats.nsmallest(5, 'similarity')
    for _, row in bottom_cross.iterrows():
        print(f"{row['pub1']} ↔ {row['pub2']}")
        print(f"  Similarity: {row['similarity']:.3f}")
        print()
    
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    avg_within = np.mean([s['avg_similarity'] for s in within_pub_stats.values()])
    avg_cross = cross_pub_stats['similarity'].mean()
    
    print(f"Average within-publication similarity: {avg_within:.3f}")
    print(f"Average cross-publication similarity:  {avg_cross:.3f}")
    print(f"Difference (coherence - alignment):    {avg_within - avg_cross:.3f}")
    print()
    
    if avg_within > avg_cross:
        print("→ Publications are more internally coherent than mutually aligned")
        print("  (Each has distinct identity despite some shared perspectives)")
    else:
        print("→ Publications are highly aligned across the network")
        print("  (Strong ideological cohesion across different publications)")
    
    return avg_within, avg_cross

def save_embeddings_and_similarity(embeddings_df, similarity_matrix, pub_sim_df, 
                                   within_pub_stats, cross_pub_stats):
    """
    Save embeddings and similarity data to CSV files
    """
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    
    # Save page-level embeddings with page_id
    embeddings_save = embeddings_df.copy()
    embeddings_save['embedding'] = embeddings_save['embedding'].apply(lambda x: x.tolist())
    embeddings_save.to_json(OUTPUT_DIR / 'page_embeddings.json', orient='records', indent=2)
    print(f"✓ Saved page embeddings with page_id to: {OUTPUT_DIR / 'page_embeddings.json'}")
    
    # Save page-level similarity matrix (compressed numpy format)
    np.save(OUTPUT_DIR / 'page_similarity_matrix.npy', similarity_matrix)
    print(f"✓ Saved page similarity matrix to: {OUTPUT_DIR / 'page_similarity_matrix.npy'}")
    
    # Save publication-level similarity
    pub_sim_df.to_csv(OUTPUT_DIR / 'publication_similarity.csv')
    print(f"✓ Saved publication similarity to: {OUTPUT_DIR / 'publication_similarity.csv'}")
    
    # Save within-publication statistics
    within_df = pd.DataFrame([
        {
            'publication': pub,
            'avg_internal_similarity': stats['avg_similarity'],
            'n_pages': stats['n_pages'],
            'n_comparisons': stats['n_comparisons']
        }
        for pub, stats in within_pub_stats.items()
    ])
    within_df = within_df.sort_values('avg_internal_similarity', ascending=False)
    within_df.to_csv(OUTPUT_DIR / 'within_publication_coherence.csv', index=False)
    print(f"✓ Saved within-publication stats to: {OUTPUT_DIR / 'within_publication_coherence.csv'}")
    
    # Save cross-publication statistics
    cross_pub_stats.to_csv(OUTPUT_DIR / 'cross_publication_similarities.csv', index=False)
    print(f"✓ Saved cross-publication stats to: {OUTPUT_DIR / 'cross_publication_similarities.csv'}")
    
    # Save summary statistics
    summary = {
        'model': 'all-MiniLM-L6-v2',
        'embedding_dimension': 384,
        'n_pages': len(embeddings_df),
        'n_publications': embeddings_df['publication_name'].nunique(),
        'publications': embeddings_df['publication_name'].unique().tolist(),
        'pages_per_publication': embeddings_df.groupby('publication_name').size().to_dict(),
        'avg_within_pub_similarity': float(np.diag(pub_sim_df).mean()),
        'avg_cross_pub_similarity': float(pub_sim_df.values[~np.eye(len(pub_sim_df), dtype=bool)].mean()),
        'overall_avg_similarity': float(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean())
    }
    
    with open(OUTPUT_DIR / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to: {OUTPUT_DIR / 'summary_stats.json'}")
    
    print(f"\nAll data saved to: {OUTPUT_DIR}/")
    
    return summary

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
    print(f"✓ Saved comparison to: {OUTPUT_DIR / 'reuse_vs_similarity_comparison.csv'}")
    
    # Print correlation
    if len(comp_df) > 2:
        corr = comp_df['semantic_similarity'].corr(comp_df['text_reuse_count'])
        print(f"\nCorrelation between semantic similarity and text reuse: {corr:.3f}")
        print("(Positive = publications that are ideologically aligned also share text)")
        print("(Near zero = ideological alignment independent of text sharing)")

def main():
    """
    Main execution function - Dual Approach Analysis
    CSV outputs only, no visualizations
    """
    print("="*60)
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("Dual Approach: Text Reuse + Semantic Similarity")
    print("="*60)
    print("\nThis analysis complements text reuse detection by measuring")
    print("ideological alignment through semantic similarity of all pages.")
    print()
    
    # 1. Load metadata
    print("STEP 1: Loading page metadata")
    print("-" * 60)
    metadata = load_metadata('page_metadata.csv')
    
    # 2. Generate page-level embeddings with page_id
    print("\nSTEP 2: Generating page-level embeddings")
    print("-" * 60)
    embeddings_df, embeddings = generate_page_embeddings(metadata, model_name='all-MiniLM-L6-v2')
    
    # 3. Calculate page-level similarity matrix
    print("\nSTEP 3: Calculating page-level similarities")
    print("-" * 60)
    similarity_matrix = calculate_page_similarity_matrix(embeddings)
    
    # 4. Export ALL page similarities to CSV
    print("\nSTEP 4: Exporting all page-level similarities")
    print("-" * 60)
    all_similarities = export_all_page_similarities(similarity_matrix, metadata)
    
    # 5. Aggregate to publication level
    print("\nSTEP 5: Aggregating to publication level")
    print("-" * 60)
    pub_sim_df, within_pub_stats, cross_pub_stats = aggregate_to_publication_level(
        similarity_matrix, metadata
    )
    
    # 6. Analyze patterns
    print("\nSTEP 6: Analyzing similarity patterns")
    print("-" * 60)
    avg_within, avg_cross = analyze_within_vs_cross_publication(
        pub_sim_df, within_pub_stats, cross_pub_stats
    )
    
    # 7. Save all data
    print("\nSTEP 7: Saving results")
    print("-" * 60)
    summary = save_embeddings_and_similarity(
        embeddings_df, similarity_matrix, pub_sim_df, 
        within_pub_stats, cross_pub_stats
    )
    
    # 8. Compare with text reuse results
    print("\nSTEP 8: Comparing with text reuse results")
    print("-" * 60)
    create_comparison_with_text_reuse(pub_sim_df)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nModel: {summary['model']}")
    print(f"Pages analyzed: {summary['n_pages']}")
    print(f"Publications: {summary['n_publications']}")
    print(f"\nAverage within-publication similarity: {avg_within:.3f}")
    print(f"Average cross-publication similarity:  {avg_cross:.3f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated CSV files:")
    print("  - all_page_similarities.csv             (ALL pairwise page similarities)")
    print("  - publication_similarity.csv            (Publication-level aggregation)")
    print("  - within_publication_coherence.csv      (Internal coherence stats)")
    print("  - cross_publication_similarities.csv    (Cross-publication alignment)")
    print("  - reuse_vs_similarity_comparison.csv    (Comparison with text reuse)")
    print("  - page_embeddings.json                  (Page embeddings with IDs)")
    print("  - page_similarity_matrix.npy            (Full similarity matrix)")
    print("  - summary_stats.json                    (Overall statistics)")
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("\nHigh within-pub, Low cross-pub:")
    print("  → Each publication has distinct identity/focus")
    print("\nLow within-pub, High cross-pub:")
    print("  → Publications cover diverse topics but share perspectives")
    print("\nHigh within-pub, High cross-pub:")
    print("  → Strong ideological cohesion across network")
    print("\nCompare with text reuse results:")
    print("  → Do ideologically aligned pubs also share text directly?")
    print("  → Or do they develop similar views independently?")
    print("="*60)

if __name__ == "__main__":
    main()