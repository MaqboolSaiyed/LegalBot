from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import re
from functools import lru_cache
import torch
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationAgent:
    def __init__(self):
        """Initializes the SummarizationAgent with an optimized model."""
        try:
            # Using a smaller, faster model for better response time
            model_name = "google/flan-t5-base"  # Changed to base model for faster inference

            logger.info(f"Loading summarization model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Changed to float32 for CPU compatibility
                low_cpu_mem_usage=True
            )

            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)

            # Set model to evaluation mode for faster inference
            self.model.eval()

            logger.info(f"SummarizationAgent: Successfully loaded {model_name} model on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load summarization model '{model_name}': {e}")
            logger.error("Please ensure 'transformers' and a backend like 'torch' or 'tensorflow' are installed.")
            logger.error("You can install them using: pip install transformers torch")
            self.model = None
            self.tokenizer = None

    def process(self, relevant_info: str, query: str) -> str:
        """Process and format the relevant information (a single string) based on the query type."""
        if not relevant_info or not relevant_info.strip():
            logger.info("No information provided to process.")
            return "I couldn't find specific information about that in my knowledge base. Please try rephrasing your question or ask about Indian litigation or ICAI guidelines."

        query_lower = query.lower()

        # The QueryAgent now extracts a single, contiguous section, so we can directly format it
        cleaned_info = self._clean_text(relevant_info) # Still good to clean up potential extra whitespace/headers

        # Format based on query type. Prioritize formatting functions based on query keywords.
        if "steps" in query_lower and ("file" in query_lower or "lawsuit" in query_lower or "petition" in query_lower):
            return self._format_lawsuit_steps(cleaned_info)
        elif "document" in query_lower and ("required" in query_lower or "need" in query_lower or "filing" in query_lower or "paperwork" in query_lower):
            return self._format_required_documents(cleaned_info)
        elif "icai" in query_lower and ("guidelines" in query_lower or "chartered accountant" in query_lower or "ca" in query_lower or "ethics" in query_lower or "audit" in query_lower):
            return self._format_icai_guidelines(cleaned_info)
        else:
            # Fallback logic: Try to identify the section from the content itself if query keywords didn't match.
            # The content (cleaned_info) should ideally start with "## Actual Header Text".
            lines = cleaned_info.strip().split('\n')
            potential_header_line = ""
            if lines:
                potential_header_line = lines[0].strip()

            if potential_header_line.startswith("## "):
                actual_header_text = potential_header_line[3:].strip().lower() # Get text after "## "
                # Check keywords in the extracted actual_header_text
                if "steps" in actual_header_text and ("lawsuit" in actual_header_text or "file" in actual_header_text or "filing" in actual_header_text):
                    return self._format_lawsuit_steps(cleaned_info)
                elif "document" in actual_header_text and ("required" in actual_header_text or "filing" in actual_header_text):
                    return self._format_required_documents(cleaned_info)
                elif "icai" in actual_header_text and ("guidelines" in actual_header_text or "chartered accountant" in actual_header_text):
                    return self._format_icai_guidelines(cleaned_info)
                else:
                    # If a "## Header" exists but doesn't match specific types, use general formatter, preserving the header.
                    logger.info(f"Using general formatter for content starting with header: {potential_header_line}")
                    return self._format_general_response(cleaned_info, query) # _format_general_response will handle the "## Header"
            else:
                # Fallback to general response if no "## Header" is identified in the content's first line.
                logger.warning("Could not identify '## Header' in the extracted content's first line. Using general formatting.")
                return self._format_general_response(cleaned_info, query)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize the input text."""
        # Remove excessive blank lines (more than 2 consecutive newlines)
        # Corrected regex for actual newline characters
        cleaned_text = re.sub(r'\n\s*\n(\s*\n)+', '\n\n', text)
        return cleaned_text.strip()

    def _format_lawsuit_steps(self, text: str) -> str:
        """Format the steps for filing a lawsuit. Expects 'text' to start with '## Header'."""
        lines = text.strip().split('\n')
        if not lines:
            logger.warning("Empty content passed to _format_lawsuit_steps.")
            return self._format_general_response(text, "Error: Empty content for lawsuit steps")

        extracted_header_line = lines[0].strip()
        if not extracted_header_line.startswith("## "):
            logger.warning(f"_format_lawsuit_steps expected '## Header', got: {extracted_header_line}. Prepending generic header.")
            extracted_header_line = "## Steps to File a Lawsuit"
            content_for_parsing = text # Parse the whole text if header is missing
        else:
            content_for_parsing = "\n".join(lines[1:])


        # Use regex to find steps and their content
        # This pattern looks for a number followed by a period, then captures everything until the next step number or the end of the string
        step_pattern = re.compile(r'^\s*(\d+)\.\s*(.*?)(?=\n\s*\d+\.\s*|$)', re.DOTALL | re.MULTILINE)

        steps = []
        # Find all steps using the pattern
        matches = step_pattern.findall(text)

        for match in matches:
            step_number = match[0]
            step_content = match[1].strip()

            # Split the content into main line and details (bullet points)
            content_lines = step_content.split('\n')
            main_step_title = content_lines[0]
            details = [line.strip() for line in content_lines[1:] if line.strip()]

            formatted_step = f"## {step_number}. {main_step_title}"

            # Add details with proper indentation
            if details:
                formatted_step += "\n" + "\n".join(f"   {detail}" for detail in details)

            steps.append(formatted_step)

        if not steps:
            logger.warning("Could not parse steps using regex in _format_lawsuit_steps. Formatting as general response.")
            # Pass the original text (which includes or should have its header) to general response.
            return self._format_general_response(text, query="Steps to File a Lawsuit")

        return extracted_header_line + "\n\n" + "\n\n".join(steps)

    def _format_required_documents(self, text: str) -> str:
        """Format the required documents. Expects 'text' to start with '## Header'."""
        lines = text.strip().split('\n')
        if not lines:
            logger.warning("Empty content passed to _format_required_documents.")
            return self._format_general_response(text, "Error: Empty content for required documents")

        extracted_header_line = lines[0].strip()
        if not extracted_header_line.startswith("## "):
            logger.warning(f"_format_required_documents expected '## Header', got: {extracted_header_line}. Prepending generic header.")
            extracted_header_line = "## Documents Required"
            content_for_parsing = text
        else:
            content_for_parsing = "\n".join(lines[1:])


        # Similar pattern for numbered documents
        doc_pattern = re.compile(r'^\s*(\d+)\.\s*(.*?)(?=\n\s*\d+\.\s*|$)', re.DOTALL | re.MULTILINE)

        documents = []
        matches = doc_pattern.findall(text)

        for match in matches:
            doc_number = match[0]
            doc_content = match[1].strip()

            content_lines = doc_content.split('\n')
            main_doc_title = content_lines[0]
            details = [line.strip() for line in content_lines[1:] if line.strip()]

            formatted_doc = f"## {doc_number}. {main_doc_title}"

            if details:
                formatted_doc += "\n" + "\n".join(f"   {detail}" for detail in details)

            documents.append(formatted_doc)

        if not documents:
            logger.warning("Could not parse documents using regex in _format_required_documents. Formatting as general response.")
            return self._format_general_response(text, query="Required Documents")

        return extracted_header_line + "\n\n" + "\n\n".join(documents)

    def _format_icai_guidelines(self, text: str) -> str:
        """Format the ICAI guidelines. Expects 'text' to start with '## Header'."""
        lines = text.strip().split('\n')
        if not lines:
            logger.warning("Empty content passed to _format_icai_guidelines.")
            return self._format_general_response(text, "Error: Empty content for ICAI guidelines")

        extracted_header_line = lines[0].strip()
        if not extracted_header_line.startswith("## "):
            logger.warning(f"_format_icai_guidelines expected '## Header', got: {extracted_header_line}. Prepending generic header.")
            extracted_header_line = "## Key ICAI Guidelines"
            content_for_parsing = text
        else:
            content_for_parsing = "\n".join(lines[1:])


        # Pattern for numbered guidelines
        guideline_pattern = re.compile(r'^\s*(\d+)\.\s*(.*?)(?=\n\s*\d+\.\s*|$)', re.DOTALL | re.MULTILINE)

        guidelines = []
        matches = guideline_pattern.findall(text)

        for match in matches:
            guideline_number = match[0]
            guideline_content = match[1].strip()

            content_lines = guideline_content.split('\n')
            main_guideline_title = content_lines[0]
            details = [line.strip() for line in content_lines[1:] if line.strip()]

            formatted_guideline = f"## {guideline_number}. {main_guideline_title}"

            if details:
                formatted_guideline += "\n" + "\n".join(f"   {detail}" for detail in details)

            guidelines.append(formatted_guideline)

        if not guidelines:
            logger.warning("Could not parse guidelines using regex in _format_icai_guidelines. Formatting as general response.")
            return self._format_general_response(text, query="ICAI Guidelines")

        return extracted_header_line + "\n\n" + "\n\n".join(guidelines)

    def _format_general_response(self, text: str, query: str) -> str:
        """Format a general response. Uses existing '## Header' if present in 'text'."""
        input_lines = text.strip().split('\n')

        final_header = f"## Response to: {query}" # Default header
        body_lines = []

        if input_lines and input_lines[0].strip().startswith("## "):
            final_header = input_lines[0].strip() # Use existing ## Header
            # Content is from the second line onwards
            body_lines = [line.strip() for line in input_lines[1:] if line.strip()]
        else:
            # If no "## Header" in the first line, use all input lines as body
            body_lines = [line.strip() for line in input_lines if line.strip()]

        formatted_text_body = "\n".join(body_lines)
        return final_header + "\n\n" + formatted_text_body
