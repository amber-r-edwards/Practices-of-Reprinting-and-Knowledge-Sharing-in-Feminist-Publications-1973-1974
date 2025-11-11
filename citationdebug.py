#!/usr/bin/env python3
"""
Debug script to test citation removal
"""

import pandas as pd # type: ignore
import re

def remove_citation_footer(text: str) -> str:
    """
    Remove citation footer from text using multiple patterns.
    """
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    
    # Method 1: Simple split on "Archives of Sexuality and Gender"
    if 'Archives of Sexuality and Gender' in text or 'Archives of sexuality and gender' in text:
        # Find last occurrence
        idx = max(
            text.rfind('Archives of Sexuality and Gender'),
            text.rfind('Archives of sexuality and gender')
        )
        
        if idx > 0:
            # Look backwards to find the title (starts with ")
            for i in range(idx - 1, max(0, idx - 500), -1):
                if text[i] == '"':
                    return text[:i].strip()
            
            # If no quote found, just remove from "Archives" onwards
            return text[:idx].strip()
    
    # Method 2: Regex pattern for the full citation
    # Matches: "Title." Pub, Date, p. [#]. Archives ... Accessed date.
    pattern = r'"[^"]+"\.\s+[^,]+,\s+[^,]+,\s+p\.\s*\[[^\]]+\]\.\s*Archives of[^\n]+?Accessed\s+\d{1,2}\s+\w+\.?\s+\d{4}\.'
    
    text_cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    if text_cleaned != text:
        return text_cleaned.strip()
    
    # Method 3: Just remove anything after "Archives of Sexuality and Gender"
    text_cleaned = re.split(
        r'Archives of [Ss]exuality and [Gg]ender',
        text,
        maxsplit=1
    )[0].strip()
    
    # Remove trailing quote and publication info if present
    text_cleaned = re.sub(r'"[^"]+"\.\s+[^,]+,\s+[^,]+,\s+p\.\s*\[[^\]]+\]\.\s*$', '', text_cleaned)
    
    return text_cleaned.strip()

# Load your metadata
metadata = pd.read_csv('page_metadata.csv')

print("="*70)
print("CITATION REMOVAL DIAGNOSTIC")
print("="*70)

# Check 1: Do any texts contain the citation marker?
has_archives = metadata['text'].str.contains('Archives of', case=False, na=False)
print(f"\n1. Pages containing 'Archives of': {has_archives.sum()} / {len(metadata)}")

if has_archives.sum() == 0:
    print("\n⚠️  WARNING: No pages contain 'Archives of' - citations may have different format!")
    print("\nLet's check what's at the END of your text files...")
    print("\nShowing last 200 characters of first 5 pages:")
    for i in range(min(5, len(metadata))):
        print(f"\n--- Page {i} ---")
        print(metadata.iloc[i]['text'][-200:])
    print("\n" + "="*70)
    print("Copy one of the citation formats above and we'll create a custom pattern!")
    
else:
    # Show examples with citations
    print("\nShowing 3 examples with 'Archives of':")
    examples = metadata[has_archives].head(3)
    
    for idx, row in examples.iterrows():
        print(f"\n{'='*70}")
        print(f"Example {idx + 1}:")
        print(f"Publication: {row['publication_name']}")
        print(f"\nLast 300 characters of ORIGINAL text:")
        print(row['text'][-300:])
        
        # Test citation removal
        cleaned = remove_citation_footer(row['text'])
        
        print(f"\nLast 200 characters AFTER citation removal:")
        print(cleaned[-200:] if len(cleaned) > 200 else cleaned)
        
        print(f"\nOriginal length: {len(row['text'])} chars")
        print(f"After removal: {len(cleaned)} chars")
        print(f"Removed: {len(row['text']) - len(cleaned)} chars")
        
        if len(row['text']) == len(cleaned):
            print("⚠️  WARNING: Nothing was removed!")

# Check 2: Test on all pages
print("\n" + "="*70)
print("2. Testing citation removal on ALL pages...")
print("="*70)

metadata['text_no_citations'] = metadata['text'].apply(remove_citation_footer)
metadata['citation_removed'] = metadata['text'].str.len() != metadata['text_no_citations'].str.len()

print(f"\nPages where citation was removed: {metadata['citation_removed'].sum()}")
print(f"Pages where nothing changed: {(~metadata['citation_removed']).sum()}")

# Show length changes
metadata['length_diff'] = metadata['text'].str.len() - metadata['text_no_citations'].str.len()
print(f"\nAverage characters removed: {metadata['length_diff'].mean():.1f}")
print(f"Max characters removed: {metadata['length_diff'].max()}")
print(f"Min characters removed: {metadata['length_diff'].min()}")

# Check 3: Are texts too short after removal?
metadata['text_clean'] = metadata['text_no_citations'].apply(lambda x: str(x).lower().strip())
metadata['word_count'] = metadata['text_clean'].apply(lambda x: len(x.split()))

print("\n" + "="*70)
print("3. Word count analysis")
print("="*70)
print(f"\nPages with >= 20 words: {(metadata['word_count'] >= 20).sum()}")
print(f"Pages with < 20 words: {(metadata['word_count'] < 20).sum()}")
print(f"\nWord count statistics:")
print(metadata['word_count'].describe())

# Check 4: Look at pages with fewest words
print("\n" + "="*70)
print("4. Pages with FEWEST words (might be citation-only):")
print("="*70)
shortest = metadata.nlargest(5, 'length_diff')[['publication_name', 'word_count', 'length_diff']]
print(shortest)

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)