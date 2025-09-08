import os
import markdown
from markupsafe import Markup

def setup_easyocr_environment():
    """Setup EasyOCR environment variables"""
    os.environ['EASYOCR_MODULE_PATH'] = '/tmp/easyocr_models'
    print(f"EasyOCR model path set to: {os.environ.get('EASYOCR_MODULE_PATH', 'default')}")

def markdown_filter(text):
    """Convert markdown to HTML for template rendering"""
    if not text:
        return ""
    html = markdown.markdown(text)
    return Markup(html)