#!/usr/bin/env python3
"""
PDF OCR Processing with Tesseract and AI Correction
===================================================

This script processes PDF documents by:
1. Converting each page to grayscale images
2. Running Tesseract OCR on each page
3. Using OpenAI to correct OCR errors
4. Combining pages into a single text file

Usage:
    python ocr_processing.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from openai import OpenAI


# Configuration
PDF_DIR = "pdfs"
OUTPUT_DIR = "txtfiles"
TEMP_DIR_PREFIX = "ocr_temp_"


def check_dependencies():
    """Check if required packages and tools are installed."""
    missing = []
    
    # Check Python packages
    try:
        import pytesseract
        import PIL
        from pdf2image import convert_from_path
        from openai import OpenAI
    except ImportError as e:
        missing.append(str(e))
    
    # Check Tesseract
    try:
        pytesseract.get_tesseract_version()
    except:
        missing.append("Tesseract OCR not found. Install with: brew install tesseract")
    
    if missing:
        print("❌ Missing dependencies:")
        for item in missing:
            print(f"  - {item}")
        print("\nInstall missing packages with:")
        print("  pip install pytesseract pillow pdf2image openai")
        sys.exit(1)


def get_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    return api_key


def list_pdf_files():
    """List all PDF files in the PDF directory."""
    if not os.path.exists(PDF_DIR):
        print(f"❌ Directory not found: {PDF_DIR}")
        return []
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    return sorted(pdf_files)


def select_pdf(pdf_files):
    """
    Display menu and let user select a PDF file.
    
    Args:
        pdf_files (list): List of PDF filenames
    
    Returns:
        str: Selected filename or None to quit
    """
    if not pdf_files:
        print("❌ No PDF files found in the pdfs directory")
        return None
    
    print("\n" + "="*80)
    print("Available PDF Files:")
    print("="*80)
    print("0. Exit")
    
    for idx, filename in enumerate(pdf_files, 1):
        print(f"{idx}. {filename}")
    
    print("="*80)
    
    while True:
        try:
            choice = input("\nSelect a file number (or 0 to exit): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(pdf_files):
                return pdf_files[choice_num - 1]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(pdf_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def pdf_to_grayscale_images(pdf_path, temp_dir):
    """
    Convert PDF pages to grayscale images in a temporary directory.
    
    Args:
        pdf_path (str): Path to PDF file
        temp_dir (str): Temporary directory to store images
    
    Returns:
        list: Paths to grayscale image files
    """
    print(f"\n📄 Converting PDF to grayscale images...")
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        image_paths = []
        for idx, image in enumerate(images, 1):
            # Convert to grayscale
            grayscale_image = image.convert('L')
            
            # Save to temp directory
            image_filename = f"page_{idx:03d}.png"
            image_path = os.path.join(temp_dir, image_filename)
            grayscale_image.save(image_path, 'PNG')
            
            image_paths.append(image_path)
            print(f"  ✓ Page {idx}/{len(images)} converted")
        
        print(f"✅ Converted {len(images)} pages to grayscale")
        return image_paths
    
    except Exception as e:
        print(f"❌ Error converting PDF to images: {e}")
        return []


def remove_tesseract_header(text):
    """
    Remove common Tesseract header information from OCR text.
    
    Args:
        text (str): Raw OCR text
    
    Returns:
        str: Text with headers removed
    """
    lines = text.split('\n')
    
    # Skip lines that look like headers
    filtered_lines = []
    skip_patterns = [
        'OCR Results',
        'Image:',
        'Word count:',
        'Confidence:',
        '--------------------------------------------------',
        '================',
    ]
    
    for line in lines:
        # Skip empty lines at the start
        if not filtered_lines and not line.strip():
            continue
        
        # Skip header patterns
        if any(pattern in line for pattern in skip_patterns):
            continue
        
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def ocr_image(image_path):
    """
    Perform OCR on an image using Tesseract.
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        str: Extracted text
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, config='--psm 3')
        return text
    except Exception as e:
        print(f"❌ Error performing OCR on {image_path}: {e}")
        return ""


def correct_text_with_ai(text, client, page_num=None):
    """
    Use OpenAI to correct OCR errors.
    
    Args:
        text (str): Raw OCR text
        client (OpenAI): OpenAI client instance
        page_num (int): Page number for display
    
    Returns:
        str: Corrected text
    """
    page_label = f" (Page {page_num})" if page_num else ""
    print(f"  🤖 Correcting with AI{page_label}...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at correcting OCR text from historical documents. Preserve the original meaning, layout, and structure."
                },
                {
                    "role": "user",
                    "content": f"""Please correct the following OCR text from a historical document. 

IMPORTANT: Process the ENTIRE text from beginning to end. Do not stop early or truncate.

Rules:
1. Fix obvious OCR errors (like '0' instead of 'O', '1' instead of 'l')
2. Preserve the original line breaks and text blocks.
3. Preserve the original meaning and historical context
4. Do not add any additional text
5. Do not delete any text unless correcting an obvious error
6. Process EVERY line - do not skip any content

Original OCR text:
{text}

Corrected text:"""
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text
    
    except Exception as e:
        print(f"❌ Error correcting text with AI: {e}")
        return text  # Return original text if correction fails


def process_pdf(pdf_filename, api_key):
    """
    Process a PDF file: OCR + AI correction.
    
    Args:
        pdf_filename (str): Name of PDF file in pdfs directory
        api_key (str): OpenAI API key
    
    Returns:
        bool: True if successful, False otherwise
    """
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    output_filename = Path(pdf_filename).stem + ".txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print("\n" + "="*80)
    print(f"Processing: {pdf_filename}")
    print("="*80)
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    
    try:
        # Convert PDF to grayscale images
        image_paths = pdf_to_grayscale_images(pdf_path, temp_dir)
        
        if not image_paths:
            return False
        
        # Process each page
        print(f"\n🔍 Processing {len(image_paths)} pages...")
        corrected_pages = []
        
        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n[Page {idx}/{len(image_paths)}]")
            
            # Perform OCR
            print(f"  📖 Running Tesseract OCR...")
            raw_text = ocr_image(image_path)
            
            # Remove Tesseract headers
            cleaned_text = remove_tesseract_header(raw_text)
            
            if not cleaned_text.strip():
                print(f"  ⚠️  No text extracted from page {idx}")
                continue
            
            # Correct with AI
            corrected_text = correct_text_with_ai(cleaned_text, client, idx)
            corrected_pages.append(corrected_text)
            
            print(f"  ✅ Page {idx} complete")
        
        # Combine all pages
        print(f"\n📝 Combining {len(corrected_pages)} pages...")
        combined_text = '\n\n'.join(corrected_pages)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"\n✅ Saved to: {output_path}")
        print(f"📊 Total characters: {len(combined_text)}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error processing PDF: {e}")
        return False
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary files")


def main():
    """Main function with interactive menu."""
    print("="*80)
    print("PDF OCR Processing with Tesseract and AI Correction")
    print("="*80)
    
    # Check dependencies
    check_dependencies()
    
    # Get API key
    api_key = get_api_key()
    
    # Main processing loop
    while True:
        # Get list of PDF files
        pdf_files = list_pdf_files()
        
        if not pdf_files:
            break
        
        # Let user select a PDF
        selected_pdf = select_pdf(pdf_files)
        
        if selected_pdf is None:
            print("\n👋 Exiting...")
            break
        
        # Process the selected PDF
        success = process_pdf(selected_pdf, api_key)
        
        if success:
            print("\n" + "="*80)
            print("✅ Processing complete!")
        else:
            print("\n" + "="*80)
            print("⚠️  Processing encountered errors")
        
        # Ask if user wants to continue
        print("="*80)
        continue_choice = input("\nProcess another file? (yes/no): ").strip().lower()
        
        if continue_choice not in ['yes', 'y']:
            print("\n👋 Exiting...")
            break
    
    print("="*80)


if __name__ == "__main__":
    main()