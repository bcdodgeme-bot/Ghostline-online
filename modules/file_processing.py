# modules/file_processing.py - File Processing Module

import os
import io
import tempfile
from PIL import Image
import fitz
import docx
import markdown
from markupsafe import Markup
from flask import request, redirect
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, track_uploaded_file
from modules.brain import enhanced_retrieve

PROJECTS = [
    'Personal Operating Manual','AMCF','BCDodgeme','Rose and Angel','Meals N Feelz',
    'TV Signals','Damn It Carl','HalalBot','Kitchen','Health','Side Quests'
]

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

def setup_easyocr_environment():
    """Setup writable directory for EasyOCR models"""
    try:
        # Create a writable temp directory for EasyOCR
        easyocr_dir = os.path.join(tempfile.gettempdir(), 'easyocr_models')
        os.makedirs(easyocr_dir, exist_ok=True)
        
        # Set environment variable to override EasyOCR's default path
        os.environ['EASYOCR_MODULE_PATH'] = easyocr_dir
        
        print(f"EasyOCR model path set to: {easyocr_dir}")
        return True
    except Exception as e:
        print(f"Failed to setup EasyOCR environment: {e}")
        return False

def process_image_ocr(file_stream, filename):
    """Process image with EasyOCR, handling model download issues"""
    try:
        import easyocr
        import numpy as np
        
        print(f"Starting OCR processing for: {filename}")
        
        # Reset stream position
        file_stream.seek(0)
        
        # Open and convert image
        img = Image.open(file_stream)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        print(f"Image loaded: {img.size}, mode: {img.mode}")
        
        # Initialize EasyOCR with error handling for model downloads
        try:
            # Try to create reader with custom model path
            reader = easyocr.Reader(
                ['en'], 
                gpu=False,  # Disable GPU for server compatibility
                download_enabled=True,  # Allow model downloads
                model_storage_directory=os.environ.get('EASYOCR_MODULE_PATH')
            )
            print("EasyOCR reader initialized successfully")
            
        except Exception as model_error:
            print(f"EasyOCR model initialization failed: {model_error}")
            
            # Fallback: try without custom directory
            reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR reader initialized with default settings")
        
        # Perform OCR
        results = reader.readtext(img_array)
        
        if results:
            text = '\n'.join([result[1] for result in results if result[1].strip()])
            print(f"OCR extracted {len(results)} text regions, {len(text)} characters")
            return text
        else:
            print("No OCR results found")
            return "No text detected in image"
            
    except ImportError as e:
        print(f"EasyOCR not installed: {e}")
        raise Exception("EasyOCR not installed. Please install with: pip install easyocr opencv-python-headless")
    
    except Exception as e:
        print(f"OCR processing failed: {e}")
        raise Exception(f"OCR processing failed: {str(e)}")

def analyze_image_with_vision(file_stream, filename):
    """Analyze image using GPT-4 Vision when OCR results are poor"""
    try:
        import base64
        import requests
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return "OpenAI API key not configured for vision analysis"
        
        # Reset stream and encode image
        file_stream.seek(0)
        image_data = base64.b64encode(file_stream.read()).decode('utf-8')
        
        # Determine image format for data URL
        file_extension = filename.split('.')[-1].lower()
        mime_type = f"image/{file_extension}" if file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp'] else "image/jpeg"
        
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create vision analysis prompt
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this image in detail. If it contains charts, graphs, screenshots, or data visualizations, describe the key insights, trends, and important information. If it's a photo, describe what you see. Be specific and actionable in your analysis."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        print(f"Sending image to GPT-4 Vision for analysis: {filename}")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            vision_analysis = result['choices'][0]['message']['content']
            print(f"GPT-4 Vision analysis successful: {len(vision_analysis)} characters")
            return vision_analysis
        else:
            print(f"GPT-4 Vision API error: {response.status_code} - {response.text}")
            return f"Vision analysis failed: API error {response.status_code}"
            
    except Exception as e:
        print(f"Vision analysis failed: {e}")
        return f"Vision analysis error: {str(e)}"

def markdown_filter(text):
    """Convert markdown to HTML"""
    if not text:
        return ""
    # Configure markdown with basic extensions
    md = markdown.Markdown(extensions=['nl2br', 'fenced_code'])
    return Markup(md.convert(text))

def handle_file_upload():
    """Handle file upload processing"""
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file uploaded", 400
        
        # Get current project from form or session
        project = request.form.get('project', PROJECTS[0])
        filename = file.filename.lower()
        text = ""
        
        print(f"Processing file: {filename} for project: {project}")

        # Process different file types
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            text = process_image_ocr(file.stream, filename)
                
        elif filename.endswith('.pdf'):
            try:
                file.stream.seek(0)
                data = file.read()
                
                if not data:
                    return "PDF file appears to be empty", 400
                
                doc = fitz.open(stream=data, filetype="pdf")
                
                if doc.page_count == 0:
                    return "PDF has no pages", 400
                
                text_parts = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(f"=== Page {page_num + 1} ===\n{page_text}")
                
                text = "\n\n".join(text_parts)
                doc.close()
                
                if not text.strip():
                    text = "No text found in PDF (may be image-based or encrypted)"
                    
            except Exception as e:
                print(f"PDF processing failed: {e}")
                return f"PDF Error: {str(e)}", 500
                
        elif filename.endswith('.docx'):
            try:
                file.stream.seek(0)
                file_data = file.read()
                
                if not file_data:
                    return "Word document appears to be empty", 400
                
                file_stream = io.BytesIO(file_data)
                document = docx.Document(file_stream)
                
                paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
                
                tables_text = []
                for table in document.tables:
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            tables_text.append(" | ".join(row_text))
                
                all_text = []
                if paragraphs:
                    all_text.extend(paragraphs)
                if tables_text:
                    all_text.append("\n=== Tables ===")
                    all_text.extend(tables_text)
                
                text = "\n".join(all_text)
                
                if not text.strip():
                    text = "No readable text found in Word document"
                    
            except Exception as e:
                print(f"Word document processing failed: {e}")
                return f"Word Document Error: {str(e)}", 500
                
        else:
            return "Unsupported file type. Supported: PNG, JPG, JPEG, GIF, BMP, PDF, DOCX", 400

        # Truncate if too long
        if len(text) > 15000:
            text = text[:15000] + "\n\n[...Content truncated...]"
        
        # Check if OCR results are meaningful (fallback logic)
        text_words = len(text.split()) if text else 0
        is_image_file = filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        
        # Create analysis prompt based on OCR quality
        if is_image_file and text_words < 5:
            # Poor OCR results - switch to GPT-4 Vision analysis
            print(f"OCR extracted only {text_words} words, switching to vision analysis")
            
            # Get vision analysis
            vision_description = analyze_image_with_vision(file.stream, file.filename)
            
            analysis_prompt = f"""I've uploaded an image file '{file.filename}'. OCR only extracted: "{text.strip()}"

Since the text was minimal, I analyzed the image visually instead:

=== VISUAL ANALYSIS ===
{vision_description}

=== FILE DETAILS ===
- Filename: {file.filename}
- Type: IMAGE ({filename.split('.')[-1].upper()})
- Analysis Method: GPT-4 Vision (due to minimal text extraction)

Please provide insights based on this visual analysis."""

        else:
            # Good OCR results - proceed with text analysis
            analysis_prompt = f"""I've uploaded and processed the file '{file.filename}'. Here's what was extracted:

=== EXTRACTED CONTENT ===
{text}

=== FILE DETAILS ===
- Filename: {file.filename}
- Type: {filename.split('.')[-1].upper()}
- Characters: {len(text):,}
- Words: {len(text.split()):,}
- Lines: {len(text.splitlines()):,}

Please analyze this content and provide insights, summaries, or answer any questions about what you see."""

        print(f"File processing successful: {len(text)} characters extracted")

        # Get AI voices from form or use default
        use_voices = ['SyntaxPrime']  # Default to SyntaxPrime for file analysis
        random_toggle = False

        # Generate AI analysis
        try:
            retrieval_ctx = enhanced_retrieve(analysis_prompt, k=5) if is_ready() else []
            response_data = generate_response(
                analysis_prompt, use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
        except Exception as e:
            print(f"AI analysis failed: {e}")
            response_data = {"SyntaxPrime": f"File processed successfully, but AI analysis failed: {e}"}

        # Save to database and track file upload
        user_message = f"[File Upload] {file.filename}"
        save_conversation_enhanced(project, user_message, response_data)
        
        # Track the uploaded file in database
        file_extension = filename.split('.')[-1].upper() if '.' in filename else 'UNKNOWN'
        content_summary = text[:500] if text else "No text extracted"
        track_uploaded_file(file.filename, file_extension, project, content_summary)

        # Redirect back to main chat with the analysis
        return redirect(f'/?project={project}#bottom-anchor')
        
    except Exception as e:
        print(f"Upload route failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return f"Upload Error: {str(e)}", 500