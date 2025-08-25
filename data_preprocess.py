import os
from typing import List, Dict

from docx import Document
import pytesseract
import fitz
from pdf2image import convert_from_path
from typing import Generator
import pypdf
from loguru import logger


class DocumentReader:
    def __init__(self, path: str):
        self.path = path
        self.file_type = self._detect_file_type()

    def _detect_file_type(self) -> str:
        if self.path.endswith("pdf"):
            return "pdf"
        elif self.path.endswith("docx"):
            return "docx"
        else:
            raise TypeError(f"Unsupported file type for reading: {self.path}")

    def is_structured_pdf(self) -> bool:
        """Checks if a PDF file contains structured text on the first page."""
        pdf_document = fitz.open(self.path)
        first_page = pdf_document[0]
        text = first_page.get_text("text")
        pdf_document.close()
        return bool(text.strip())

    def read_pdf(self) -> Generator[str, None, None]:
        """Yields each page of structured text from a PDF."""
        with open(self.path, "rb") as pdfFileObj:
            pdf_reader = pypdf.PdfReader(pdfFileObj)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    yield text
                else:
                    yield from self.read_scan()  # Fallback to OCR if text is missing

    def read_scan(self) -> Generator[str, None, None]:
        """Yields text for each page in a scanned (image-based) PDF."""
        doc_images = convert_from_path(self.path)

        for page_image in doc_images:
            try:
                osd_info = pytesseract.image_to_osd(page_image)
                rotation_angle = int(osd_info.split("Rotate: ")[1].split("\n")[0])
                if rotation_angle == 180:
                    page_image = page_image.rotate(rotation_angle, expand=True)
                text = pytesseract.image_to_string(page_image, lang="pol")
                yield text
            except pytesseract.TesseractError as e:
                logger.warning(e)
                continue

    def read_docx(self) -> Generator[str, None, None]:
        """Yields each paragraph from a DOCX file as a separate page."""
        doc = Document(self.path)
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                yield paragraph.text

    def read_next_page(self) -> Generator[str, None, None]:
        """Determines file type and yields one page of text at a time."""
        if self.file_type == "pdf":
            if self.is_structured_pdf():
                yield from self.read_pdf()
            else:
                yield from self.read_scan()

        elif self.file_type == "docx":
            yield from self.read_docx()

    def read_all_pages_list(self) -> List[str]:
        """Reads all pages and returns a list of strings, one per page."""
        return list(self.read_next_page())

    def read_all_pages_string(self) -> str:
        """Reads all pages and a string containing whole document."""
        text = ""
        pages = self.read_next_page()
        for page in pages:
            text += page
        return text


DOCUMENTS_PATH = "data/processed/documents/"
SUMMARIES_PATH = "data/summaries/"
PROCESSED_DIR = "data/processed/"


def save_as_text(text: str, out_file: str):
    if os.path.exists(out_file):
        return
    with open(file=out_file, mode="w") as f:
        f.write(text)


def convert_docs_to_text_pages(pages: int):
    """

    Args:
        pages (int): How many pages of each document to convert
    """
    for doc_name in os.listdir(DOCUMENTS_PATH):
        file_path = os.path.join(DOCUMENTS_PATH + doc_name)
        doc_reader = DocumentReader(file_path)
        text = doc_reader.read_all_pages_string()

        raw_name = doc_name.split(".")[0]
        out_file = os.path.join(PROCESSED_DIR + f"documents/{raw_name}.txt")
        save_as_text(text=text, out_file=out_file)


def convert_docs_to_text():
    for doc_name in os.listdir(DOCUMENTS_PATH):
        file_path = os.path.join(DOCUMENTS_PATH + doc_name)
        doc_reader = DocumentReader(file_path)
        text = doc_reader.read_all_pages_string()

        raw_name = doc_name.split(".")[0]
        out_file = os.path.join(PROCESSED_DIR + f"documents/{raw_name}.txt")
        save_as_text(text=text, out_file=out_file)


def create_train_data_for_prompt_tuning(
    documents_path: str, target_path: str, max_len: int
):
    train_data: List[Dict[str, str]] = []
    for text_doc_name in os.listdir(documents_path):
        text_doc_path = os.path.join(documents_path, text_doc_name)
        summary_path = os.path.join(target_path, text_doc_name)
        # find summary
        if not os.path.exists(summary_path):
            print(f"{text_doc_path} has no summary - {summary_path}")
            continue

        text_doc = ""
        summ = ""
        with open(file=text_doc_path, mode="r") as f:
            text_doc = f.read()[:max_len]

        with open(summary_path, "r") as f:
            summ = f.read()

        train_data.append({"input": text_doc, "target": summ})
    return train_data


def get_doc_text(path: str):
    with open(file=path, mode="r") as f:
        text = f.read()

    return text


if __name__ == "__main__":
    convert_docs_to_text()
