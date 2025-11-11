# Text Analysis Project: Exploring Text Reuse in 1970s Feminist Publications

This project analyzes patterns of text reuse across four feminist publications from 1973-1974: *Ain't I A Woman*, *Big Mamma Rag*, *Do it NOW*, and *Gold Flower*. The analysis tracks direct textual sharing between publications to understand information networks and reprinting practices in the feminist underground press.

## Project Overview

This computational text analysis project employs digital humanities methods to examine how content circulated among feminist publications. The analysis focuses on identifying and visualizing patterns of text reuse, from direct reprints to modified passages, providing insights into the collaborative and networked nature of feminist publishing. As a small scale case study, one goal of this project was to generate scripts that could be reproduced to handle a larger corpus of publications across a longer period of time.

### Methods

- **OCR Processing**: Converts PDF scans to machine-readable text using Tesseract OCR engine with AI-assisted correction via OpenAI's GPT-4o-mini model for post-processing cleanup
- **Text Reuse Detection**: Identifies shared text passages using Python's `difflib.SequenceMatcher` algorithm with configurable similarity thresholds and word-based matching
- **Network Analysis**: Creates directed graphs using NetworkX library to visualize relationships between publications and individual pages based on shared content
- **Temporal Analysis**: Tracks directionality and time lag in text sharing patterns using pandas datetime processing
- **Semantic Analysis**: Generates document embeddings using sentence-transformers library with 'all-MiniLM-L6-v2' model and calculates cosine similarity via scikit-learn
- **Visualization**: Creates networks (NetworkX + Matplotlib), heatmaps (Seaborn), and temporal plots (Matplotlib) with publication-based color coding

### Key Parameters
- Minimum match length: 6 words
- Similarity threshold: 0.65 (allows for small variations in OCR and reprinting)

## Project Structure

```
TextAnalysisProject/
├── README.md                           # This file
├── .gitignore                          # Excludes PDFs, virtual environment, debug files
├── venv/                               # [IGNORED] Python virtual environment
├── pdfs/                               # [IGNORED] Original PDF source files - obtained from GALE Herstory archive
├── txtfiles/                           # Processed text files and metadata - subdirectory for each publication
│   ├── AintIAWoman/                    # Feminist zine published by the Ain't I A Woman Collective of Iowa City, Iowa
│   ├── BigMammaRag/                    # Feminist zine published out of Denver, CO
|   ├── DoItNow/                        # Monthly Newsletter of the National Organization for Women
|   └── GoldFlower/                     # Feminist zine published out of Minneapolis-St. Paul, Minnesota
├── reuse_results/                      # Text reuse analysis outputs
│   ├── text_reuse_results.csv          # All detected matches
│   ├── text_reuse_filtered.csv         # Matches excluding boilerplate
│   └── text_reuse_for_review.csv       # Sample for manual verification
├── reuse_visualizations/               # Text reuse visualizations
│   ├── network_publications_count.png  # Publication-level network generated from text_reuse_filtered.csv
│   ├── page_network.png                # Page-level network with page_IDs (sortraceable to page_metadata.csv)
│   ├── temporal_reuse.png              # Timeline of text sharing
│   ├── top_50_pages_heatmap.png        # Most connected pages heatmap
│   └── page_level_heatmap.png          # Full page similarity matrix
├── semantic_results/                   # Semantic similarity outputs (experimental)
├── ocr_processing.py                   # PDF to text conversion
├── metadatacsv.py                      # Generates page metadata
├── textreuse.py                        # Text reuse detection
├── visualize_textreuse.py              # Text reuse visualizations
├── semanticsimilarity.py               # Semantic analysis (experimental)
├── visualize_semanticsimilarity.py     # Semantic visualizations (experimental)
├── requirements.txt                    # required packages for replication
└── debug.py                            # [IGNORED] Development utilities
```

## Copyright Notice

**Important**: The original PDF files of the feminist publications are not included in this repository due to copyright restrictions. Users must obtain their own copies of the source materials before running the OCR processing pipeline. The analysis scripts and methodological framework are provided for research and educational purposes.

## Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Additional Requirements
- Tesseract OCR engine
- Python 3.8+
- Sufficient disk space for text processing outputs

## Usage Instructions

Follow these steps in order to replicate the analysis:

### 1. OCR Processing
```bash
python ocr_processing.py
```
- Converts PDF files to individual page text files
- Requires source PDFs in the `pdfs/` directory
- Outputs text files to `txtfiles/` and respective directories for each publication
- Includes OCR error correction and text cleaning

### 2. Metadata Generation
```bash
python metadatacsv.py
```
- Creates master metadata file linking all pages
- Generates unique page_id for cross-referencing
- Extracts publication names, dates, and file paths
- Creates `page_metadata.csv` used by all subsequent analyses

### 3. Text Reuse Analysis
```bash
python textreuse.py
```
- Detects shared text passages between publications
- Filters out citations and boilerplate text
- Categorizes matches by type and calculates similarity metrics
- Outputs results to `reuse_results/` directory

### 4. Text Reuse Visualizations
```bash
python visualize_textreuse.py
```
- Creates network diagrams showing text sharing relationships
- Generates heatmaps of page-level and publication-level connections
- Produces temporal visualization of sharing patterns
- Saves all visualizations to `reuse_visualizations/` directory

## Experimental Methods (Results Excluded)

### Semantic Similarity Analysis
```bash
python semanticsimilarity.py          # Generate semantic embeddings
python visualize_semanticsimilarity.py # Create semantic visualizations
```
**Note**: While originally intended as a dual approach to complement literal text reuse with ideological similarity analysis, the semantic similarity method proved ineffective for this dataset. The high variability in zine content at both page and publication levels, combined with the small scale and already-assumed ideological alignment of these publications, made semantic analysis less historically meaningful. The scripts and results are preserved here for methodological transparency, but findings from semantic analysis are excluded from the main results.

## Contact

For questions about methodology or to report issues, please open a GitHub issue or contact the project maintainer.