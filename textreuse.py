import pandas as pd # type: ignore
import numpy as np
from difflib import SequenceMatcher
from datetime import datetime
import re
from pathlib import Path
import json

# Load your metadata
metadata = pd.read_csv('page_metadata.csv')

# Add page_id if you don't have one
page_id_added = False
if 'page_id' not in metadata.columns:
    metadata['page_id'] = range(len(metadata))
    page_id_added = True

# Convert issue_date to datetime for temporal analysis
metadata['issue_date'] = pd.to_datetime(metadata['issue_date'])

# Sort by date (important for directionality)
metadata = metadata.sort_values('issue_date').reset_index(drop=True)

# Save if we added page_id
if page_id_added:
    metadata.to_csv('page_metadata.csv', index=False)
    print("✅ Added page_id column to page_metadata.csv")

print(f"Loaded {len(metadata)} pages from {metadata['publication_name'].nunique()} publications")
print(f"Date range: {metadata['issue_date'].min()} to {metadata['issue_date'].max()}")

def preprocess_text(text):
    """
    Light preprocessing to improve matching without losing content
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Normalize whitespace (but keep paragraph breaks)
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    
    # Optional: Remove common OCR artifacts
    # text = text.replace('|', 'l')  # Common OCR confusion
    # text = text.replace('0', 'o')  # If you see this pattern
    
    return text.strip()

# Add preprocessed column
metadata['text_clean'] = metadata['text'].apply(preprocess_text)

# Calculate text length for later use
metadata['text_length'] = metadata['text_clean'].apply(len)

def find_text_matches(text1, text2, min_words=8, similarity_threshold=0.85):
    """
    Find matching passages between two texts using sequence matching
    
    Parameters:
    - text1, text2: texts to compare
    - min_words: minimum length of match in words
    - similarity_threshold: minimum similarity ratio (0-1)
    
    Returns:
    - List of match dictionaries
    """
    if not text1 or not text2:
        return []
    
    # Split into words for matching
    words1 = text1.split()
    words2 = text2.split()
    
    if len(words1) < min_words or len(words2) < min_words:
        return []
    
    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, words1, words2)
    matches = []
    
    for match in matcher.get_matching_blocks():
        i, j, size = match
        
        # Check if match meets minimum word count
        if size >= min_words:
            # Extract matching text
            matched_words = words1[i:i+size]
            matched_text = ' '.join(matched_words)
            
            # Calculate similarity ratio for this segment
            segment1 = ' '.join(words1[max(0, i-5):i+size+5])  # Context window
            segment2 = ' '.join(words2[max(0, j-5):j+size+5])
            local_similarity = SequenceMatcher(None, segment1, segment2).ratio()
            
            if local_similarity >= similarity_threshold:
                matches.append({
                    'matched_text': matched_text,
                    'match_length_words': size,
                    'match_length_chars': len(matched_text),
                    'similarity_score': local_similarity,
                    'position_text1': i,
                    'position_text2': j
                })
    
    return matches

# Alternative: Character-based matching for shorter passages
def find_text_matches_chars(text1, text2, min_chars=100, similarity_threshold=0.85):
    """
    Character-based matching - useful for shorter passages or poetry
    """
    if not text1 or not text2 or len(text1) < min_chars or len(text2) < min_chars:
        return []
    
    matcher = SequenceMatcher(None, text1, text2)
    matches = []
    
    for match in matcher.get_matching_blocks():
        i, j, size = match
        
        if size >= min_chars:
            matched_text = text1[i:i+size]
            
            # Get context for similarity check
            context1 = text1[max(0, i-50):i+size+50]
            context2 = text2[max(0, j-50):j+size+50]
            local_similarity = SequenceMatcher(None, context1, context2).ratio()
            
            if local_similarity >= similarity_threshold:
                matches.append({
                    'matched_text': matched_text,
                    'match_length_chars': size,
                    'match_length_words': len(matched_text.split()),
                    'similarity_score': local_similarity,
                    'position_text1': i,
                    'position_text2': j
                })
    
    return matches

def compare_all_pages(metadata, min_words=8, similarity_threshold=0.85, 
                      same_pub=False, max_time_gap_days=None):
    """
    Compare all pairs of pages and find text reuse
    
    Parameters:
    - metadata: DataFrame with page data
    - min_words: minimum match length
    - similarity_threshold: minimum similarity
    - same_pub: if False, only compare across publications
    - max_time_gap_days: if set, only compare pages within this time window
    """
    results = []
    total_comparisons = 0
    
    # Only compare pages where source comes before or same time as target
    for idx1, row1 in metadata.iterrows():
        for idx2, row2 in metadata.iterrows():
            # Skip if same page
            if idx1 == idx2:
                continue
            
            # Only compare if row1 is earlier or same date
            if row1['issue_date'] > row2['issue_date']:
                continue
            
            # Filter by publication if requested
            if not same_pub and row1['publication_name'] == row2['publication_name']:
                continue
            
            # Filter by time gap if requested
            if max_time_gap_days:
                time_gap = (row2['issue_date'] - row1['issue_date']).days
                if time_gap > max_time_gap_days:
                    continue
            
            total_comparisons += 1
            
            # Find matches
            matches = find_text_matches(
                row1['text_clean'], 
                row2['text_clean'],
                min_words=min_words,
                similarity_threshold=similarity_threshold
            )
            
            # Record each match
            for match in matches:
                results.append({
                    'source_page_id': row1['page_id'],
                    'target_page_id': row2['page_id'],
                    'source_publication': row1['publication_name'],
                    'target_publication': row2['publication_name'],
                    'source_date': row1['issue_date'],
                    'target_date': row2['issue_date'],
                    'time_lag_days': (row2['issue_date'] - row1['issue_date']).days,
                    'source_volume': row1.get('volume', ''),
                    'source_number': row1.get('number', ''),
                    'target_volume': row2.get('volume', ''),
                    'target_number': row2.get('number', ''),
                    'source_text_length': row1['text_length'],
                    'target_text_length': row2['text_length'],
                    **match  # Unpack match details
                })
        
        # Progress indicator
        if (idx1 + 1) % 10 == 0:
            print(f"Processed {idx1 + 1}/{len(metadata)} pages...")
    
    print(f"Total comparisons: {total_comparisons}")
    print(f"Matches found: {len(results)}")
    
    return pd.DataFrame(results)

# Run the comparison
print("Starting text reuse detection...")
reuse_results = compare_all_pages(
    metadata, 
    min_words=8,
    similarity_threshold=0.85,
    same_pub=False  # Only cross-publication for now
)

# Save results
reuse_results.to_csv('text_reuse_results.csv', index=False)
print(f"Saved {len(reuse_results)} matches to text_reuse_results.csv")

def calculate_match_metrics(reuse_df):
    """
    Add additional metrics to characterize matches
    """
    # Match percentage of source and target
    reuse_df['match_pct_source'] = (
        reuse_df['match_length_chars'] / reuse_df['source_text_length'] * 100
    )
    reuse_df['match_pct_target'] = (
        reuse_df['match_length_chars'] / reuse_df['target_text_length'] * 100
    )
    
    # Categorize by length
    def categorize_length(words):
        if words >= 100:
            return 'very_long'
        elif words >= 50:
            return 'long'
        elif words >= 20:
            return 'medium'
        else:
            return 'short'
    
    reuse_df['match_length_category'] = reuse_df['match_length_words'].apply(categorize_length)
    
    return reuse_df

reuse_results = calculate_match_metrics(reuse_results)

def classify_match_type(row):
    """
    Automatically classify the type of text reuse
    """
    similarity = row['similarity_score']
    match_pct = max(row['match_pct_source'], row['match_pct_target'])
    match_words = row['match_length_words']
    
    # Full or substantial reprint
    if match_pct > 80 and similarity > 0.95:
        return 'full_reprint'
    elif match_pct > 30 and similarity > 0.90:
        return 'substantial_reprint'
    
    # Modified reprint
    elif match_words > 50 and similarity > 0.85:
        return 'modified_reprint'
    
    # Quoted passage
    elif match_words >= 15 and similarity > 0.80:
        return 'quoted_passage'
    
    else:
        return 'possible_reuse'

reuse_results['match_type_auto'] = reuse_results.apply(classify_match_type, axis=1)

# Summary of match types
print("\nMatch type distribution:")
print(reuse_results['match_type_auto'].value_counts())

def identify_boilerplate(reuse_df, min_occurrences=3):
    """
    Flag matches that appear too frequently (likely boilerplate)
    """
    # Count how many times each matched text appears
    text_counts = reuse_df['matched_text'].value_counts()
    boilerplate_texts = text_counts[text_counts >= min_occurrences].index
    
    # Flag boilerplate
    reuse_df['is_boilerplate'] = reuse_df['matched_text'].isin(boilerplate_texts)
    
    print(f"\nIdentified {len(boilerplate_texts)} potential boilerplate passages")
    print(f"Flagged {reuse_df['is_boilerplate'].sum()} matches as boilerplate")
    
    return reuse_df

reuse_results = identify_boilerplate(reuse_results)

# Create filtered version without boilerplate
reuse_meaningful = reuse_results[~reuse_results['is_boilerplate']].copy()
reuse_meaningful.to_csv('text_reuse_filtered.csv', index=False)

def create_review_file(reuse_df, metadata, output_file='text_reuse_for_review.csv', sample_size=50):
    """
    Create a file with context for manual review
    """
    # Sample diverse matches
    sample = reuse_df.groupby('match_type_auto').apply(
        lambda x: x.sample(min(len(x), sample_size // reuse_df['match_type_auto'].nunique()))
    ).reset_index(drop=True)
    
    # Add context from original texts
    review_data = []
    for _, row in sample.iterrows():
        source_text = metadata.loc[metadata['page_id'] == row['source_page_id'], 'text_clean'].iloc[0]
        target_text = metadata.loc[metadata['page_id'] == row['target_page_id'], 'text_clean'].iloc[0]
        
        # Get context (50 chars before and after)
        match_pos_source = row['position_text1']
        match_text_words = row['matched_text'].split()
        
        # Find the match in source text to get character position
        match_start = source_text.find(row['matched_text'][:50])  # Find first part
        if match_start != -1:
            context_start = max(0, match_start - 200)
            context_end = min(len(source_text), match_start + len(row['matched_text']) + 200)
            source_context = source_text[context_start:context_end]
        else:
            source_context = row['matched_text']
        
        # Similar for target
        match_start_target = target_text.find(row['matched_text'][:50])
        if match_start_target != -1:
            context_start = max(0, match_start_target - 200)
            context_end = min(len(target_text), match_start_target + len(row['matched_text']) + 200)
            target_context = target_text[context_start:context_end]
        else:
            target_context = row['matched_text']
        
        review_data.append({
            **row.to_dict(),
            'source_context': source_context,
            'target_context': target_context,
            'verified_match_type': '',  # Empty column for manual input
            'notes': ''  # Empty column for manual notes
        })
    
    review_df = pd.DataFrame(review_data)
    review_df.to_csv(output_file, index=False)
    print(f"Created review file with {len(review_df)} samples: {output_file}")

create_review_file(reuse_meaningful, metadata)