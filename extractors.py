import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted tags
        for tag in soup(["script", "style", "aside", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # Extract paragraph content
        paragraphs = soup.find_all("p")
        if not paragraphs:
            return soup.get_text(separator="\n").strip()

        texts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
        return "\n".join(texts)
    
    except Exception as e:
        print(f"[ERROR - URL Extraction] {e}")
        return ""

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    Accepts either a file path or a Streamlit UploadedFile object.
    """
    try:
        # Streamlit file upload
        if hasattr(file, "read"):
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                return "\n".join([page.get_text().strip() for page in doc])
        else:
            # File path
            with fitz.open(file) as doc:
                return "\n".join([page.get_text().strip() for page in doc])
    
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return ""
