import os
import logging
from pathlib import Path
import fitz  # PyMuPDF
import nltk
from typing import List, Dict, Optional
import requests
from tqdm import tqdm
import re
import pytesseract
from PIL import Image
import io
from nltk.tokenize import sent_tokenize
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to inject Markdown headers based on heuristics
def inject_markdown_headers(text: str) -> str:
    import re
    lines = text.split('\n')
    processed_lines = []
    # Regex to identify potential headers:
    # - Starts with an uppercase letter, followed by words (Title Case like).
    # - Or, consists of all uppercase letters and possibly numbers/spaces.
    # - Relatively short (e.g., 1 to 10 words).
    # - Does not end with a period, comma, or semicolon (less likely to be a full sentence).
    # - Is not already a markdown header.
    # - Must contain at least one letter.
    header_pattern = re.compile(
        r"^(?!\s*#)"  # Not already a markdown header
        r"([A-Z][a-zA-Z0-9\s',-]*[a-zA-Z0-9]|[A-Z0-9\s',-]+)"  # Title case or ALL CAPS
        r"$"
    )

    for line in lines:
        stripped_line = line.strip()
        if stripped_line and \
           len(stripped_line.split()) < 12 and \
           len(stripped_line) < 100 and \
           not stripped_line.endswith(('.', ',', ';', ':', '?','!')) and \
           header_pattern.match(stripped_line) and \
           any(c.isalpha() for c in stripped_line):
               processed_lines.append(f"## {stripped_line}")
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)

# Constants
DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# List of URLs for the legal documents (PDFs)
DOCUMENT_URLS = [
    "https://www.cyrilshroff.com/wp-content/uploads/2020/09/Guide-to-Litigation-in-India.pdf",
    "https://kb.icai.org/pdfs/PDFFile5b28c9ce64e524.54675199.pdf"
]

# Sample content for testing
SAMPLE_CONTENT = {
    "litigation_guide": """
    Steps to File a Lawsuit in India:

    1. Draft the Plaint
       - Prepare a detailed plaint containing:
         * Facts of the case
         * Grounds for filing the suit
         * Relief sought
         * Supporting documents
       - File at the appropriate court based on:
         * Jurisdiction (territorial and pecuniary)
         * Value of the suit
         * Type of case (civil/criminal)

    2. File the Plaint
       - Submit the plaint to the court registry
       - Pay the required court fees (based on suit value)
       - Get acknowledgment receipt
       - Keep copies of all filed documents
       - Note the case number for future reference

    3. Court Scrutiny
       - Court examines plaint for:
         * Compliance with legal requirements
         * Proper court fees
         * Jurisdiction
       - Court may:
         * Ask for additional documents
         * Suggest amendments
         * Return the plaint for corrections
         * Accept the plaint for further proceedings

    4. Summons
       - Court issues summons to defendant
       - Must be served within 30 days
       - Defendant gets 30 days to appear
       - If not served, court may:
         * Extend time for service
         * Order substituted service
         * Dismiss the case

    5. Written Statement
       - Defendant must file within 30 days
       - Can request extension (up to 90 days)
       - Must include:
         * Response to each allegation
         * Counter-claims if any
         * Supporting documents
         * List of witnesses

    6. Framing of Issues
       - Court frames issues for trial
       - Based on plaint and written statement
       - Both parties can suggest issues
       - Issues determine scope of trial

    7. Evidence
       - Plaintiff presents evidence first
       - Defendant follows
       - Each party can:
         * Present documents
         * Call witnesses
         * Cross-examine
       - Court records all evidence

    8. Arguments
       - Both parties present final arguments
       - Based on evidence and law
       - Can submit written arguments
       - Court may ask questions

    9. Judgment
       - Court delivers judgment
       - Contains:
         * Findings on each issue
         * Reasons for decision
         * Relief granted
         * Costs awarded

    10. Appeal
        - Can be filed within 30 days
        - Must show:
          * Error in judgment
          * New evidence
          * Legal grounds
        - Higher court reviews:
          * Facts
          * Law
          * Procedure
    """,

    "icai_guidelines": """
    Key ICAI Guidelines for Chartered Accountants:

    1. Professional Ethics
       - Independence:
         * Maintain objectivity
         * Avoid conflicts of interest
         * Disclose any relationships
         * No financial interest in client
       - Integrity:
         * Honest and straightforward
         * No misleading information
         * No false statements
         * Maintain professional standards

    2. Audit Standards
       - Planning:
         * Understand client's business
         * Assess risks
         * Plan audit procedures
         * Document audit strategy
       - Execution:
         * Follow SA standards
         * Gather sufficient evidence
         * Document all findings
         * Maintain working papers
       - Reporting:
         * Clear audit opinion
         * Disclose all material matters
         * Follow reporting standards
         * Include all required statements

    3. Quality Control
       - Policies:
         * Establish quality control system
         * Monitor compliance
         * Regular reviews
         * Document procedures
       - Training:
         * Regular technical updates
         * Professional development
         * Ethics training
         * Industry knowledge

    4. Documentation
       - Working Papers:
         * Complete audit trail
         * All evidence documented
         * Clear conclusions
         * Proper organization
       - Retention:
         * Keep records for 7 years
         * Secure storage
         * Easy retrieval
         * Confidentiality maintained
    """
}

def download_file(url: str, filename: str) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def download_documents(force_redownload: bool = False) -> None:
    """Download required documents if they don't exist."""
    # Use fixed names for simplicity since URLs are stable
    fixed_doc_map = {
        "litigation_guide": DOCUMENT_URLS[0] if len(DOCUMENT_URLS) > 0 else None,
        "icai_guidelines": DOCUMENT_URLS[1] if len(DOCUMENT_URLS) > 1 else None
    }

    for doc_name, url in fixed_doc_map.items():
        if not url:
            logger.warning(f"URL for {doc_name} is not defined in DOCUMENT_URLS.")
            continue
            
        filename = str(DOCUMENTS_DIR / f"{doc_name}.pdf")
        if not os.path.exists(filename) or force_redownload:
            logger.info(f"Downloading {doc_name}...")
            if not download_file(url, filename):
                logger.error(f"Failed to download {doc_name}")
                # Create a placeholder file with sample content
                if doc_name in SAMPLE_CONTENT:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(SAMPLE_CONTENT[doc_name])
                    logger.info(f"Created placeholder file for {doc_name} with sample content")
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Placeholder for {doc_name}\n")
                        f.write("This is a placeholder file. Please replace with actual document.")

def create_sample_files():
    """Create sample text files for testing."""
    try:
        for doc_name, content in SAMPLE_CONTENT.items():
            txt_file = DOCUMENTS_DIR / f"{doc_name}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created sample file: {txt_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating sample files: {e}")
        return False

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        # First try to open and read the PDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # If no text was extracted, try OCR
        if not text.strip():
            logger.info(f"No text extracted from {pdf_path}, attempting OCR...")
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"
            doc.close()
        
        # If still no text, use sample content as fallback
        if not text.strip():
            doc_name = pdf_path.stem
            if doc_name in SAMPLE_CONTENT:
                logger.info(f"Using sample content for {doc_name} as fallback")
                return SAMPLE_CONTENT[doc_name]
            else:
                logger.warning(f"No text extracted and no sample content available for {doc_name}")
                return ""
                
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        # Try to use sample content as fallback
        doc_name = pdf_path.stem
        if doc_name in SAMPLE_CONTENT:
            logger.info(f"Using sample content for {doc_name} as fallback after error")
            return SAMPLE_CONTENT[doc_name]
        return ""

def extract_text_from_pdfs(clean_existing: bool = False) -> Dict[str, str]:
    """Extract text from all PDFs in the documents directory."""
    if clean_existing:
        for file in DOCUMENTS_DIR.glob("*.txt"):
            file.unlink()

    extracted_texts = {}
    for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
        txt_file = pdf_file.with_suffix('.txt')

        if not txt_file.exists() or clean_existing:
            logger.info(f"Extracting text from {pdf_file.name}...")
            text = extract_text_from_pdf(pdf_file) # This gets raw text

            if text:
                processed_text = inject_markdown_headers(text) # Inject headers
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(processed_text) # Save processed text
                extracted_texts[pdf_file.stem] = processed_text
            else:
                logger.warning(f"No text extracted from {pdf_file.name}")
        else:
            with open(txt_file, 'r', encoding='utf-8') as f:
                extracted_texts[pdf_file.stem] = f.read()

    return extracted_texts

def get_document_text(doc_name: str) -> Optional[str]:
    """Get text content of a specific document."""
    try:
        txt_file = DOCUMENTS_DIR / f"{doc_name}.txt"
        if not txt_file.exists():
            # If file doesn't exist, create it from sample content
            if doc_name in SAMPLE_CONTENT:
                create_sample_files()
            else:
                logger.warning(f"Document {doc_name} not found and no sample content available")
                return None

        with open(txt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading document {doc_name}: {e}")
        return None

def initialize_documents() -> None:
    """Initialize documents and create text files."""
    try:
        # Create sample files
        if not create_sample_files():
            raise Exception("Failed to create sample files")

        logger.info("Document initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing documents: {e}")
        raise

def extract_with_pymupdf(pdf_path, max_pages=10):
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text += page.get_text()
    doc.close()
    gc.collect()
    return text

def process_legal_text(text, source_filename):
    """Process extracted text to improve quality for legal documents."""
    print(f"Processing extracted text from {source_filename}...")

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix common OCR/extraction issues
    text = text.replace('•', '* ')
    text = re.sub(r'([a-z])\s+([.,;:])', r'\1\2', text)  # Fix separated punctuation

    # Split into sentences for better processing
    sentences = sent_tokenize(text)

    # Reconstruct text with proper line breaks
    structured_text = "\n".join(sentences)

    # Add document source information
    structured_text = f"SOURCE: {source_filename}\n\n{structured_text}"

    # For litigation guide, try to identify and mark sections
    if "Guide-to-Litigation-in-India" in source_filename:
        # Look for potential section headers (capitalized phrases followed by numbers or periods)
        structured_text = re.sub(r'([A-Z][A-Z\s]{3,}:?)', r'\n\n\1', structured_text)

        # Try to identify document requirements sections
        if "documents required" in structured_text.lower() or "filing requirements" in structured_text.lower():
            structured_text = re.sub(r'(documents required|filing requirements|required documents)(.+?)\n\n',
                                    r'\n\nDOCUMENT REQUIREMENTS:\2\n\n',
                                    structured_text,
                                    flags=re.IGNORECASE|re.DOTALL)

    return structured_text

def normalize_headers(text):
    import re
    header_patterns = [
        r"key\s*icai\s*guidelines\s*for\s*chartered\s*accountants\s*[:\-]?",
        r"documents\s*required\s*for\s*filing\s*a\s*case\s*[:\-]?",
        r"steps\s*to\s*file\s*a\s*lawsuit\s*[:\-]?",
        r"introduction\s*[:\-]?",
        r"conclusion\s*[:\-]?",
        # Add more patterns as needed
    ]
    def replace_header(match):
        header = match.group(0)
        header = re.sub(r"\s+", " ", header.strip()).rstrip(':').title() + ":"
        return "\n" + header + "\n"
    for pattern in header_patterns:
        text = re.sub(pattern, replace_header, text, flags=re.IGNORECASE)
    return text

def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[tuple[str, str]]:
    """Split text into overlapping chunks and tag them with their section header."""
    sections = []
    current_section_content = []
    current_header = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for section headers
        section_markers = {
            'litigation_guide': 'Steps to File a Lawsuit in India:',
            'icai_guidelines': 'Key ICAI Guidelines for Chartered Accountants:',
            # Add other major section markers if needed
        }

        is_header = False
        for doc_type, marker in section_markers.items():
            if line.startswith(marker):
                # Save previous section if exists
                if current_section_content:
                    sections.append((current_header, '\n'.join(current_section_content)))
                # Start new section
                current_header = marker # Use the marker as the header tag
                current_section_content = [line] # Include the header in the first line of the content
                is_header = True
                break

        if not is_header:
             if current_header is None: # Handle text before the first header
                 current_section_content.append(line)
             else:
                 current_section_content.append(line)

    # Add the last section
    if current_section_content:
        sections.append((current_header, '\n'.join(current_section_content)))

    # Process each section into chunks with header tag
    chunks = []
    for header, content in sections:
        if not content:
            continue

        # Split content into words
        words = content.split()

        # Create chunks with overlap
        for i in range(0, len(words), chunk_size // 2):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk = ' '.join(chunk_words)
                # Attach the header tag to each chunk
                chunks.append((header, chunk))

    return chunks

# This _split_into_chunks method is specific to QueryAgent's logic and should reside there.
# Removing it from here to avoid confusion, assuming it's correctly implemented in QueryAgent.

if __name__ == '__main__':
    # This is for testing the module directly
    initialize_documents()
    for doc_name in ["litigation_guide", "icai_guidelines"]:
        content = get_document_text(doc_name)
        if content:
            print(f"\nSample of {doc_name} content (first 500 chars):")
            print(content[:500] + "...")
        else:
            print(f"\nNo content available for {doc_name}")
