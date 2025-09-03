# modules/file_processing.py - UPDATED for integrated chat flow
# Fixes Entry #4: Now keeps file analysis in the conversational thread

import os
import io
import tempfile
from PIL import Image
import fitz
import docx
import markdown
from markupsafe import Markup
from flask import request, redirect, jsonify, current_app
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

def process_single_file(file, project):
    """Process a single file and return analysis data - NEW FUNCTION"""
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
                raise Exception("PDF file appears to be empty")
            
            doc = fitz.open(stream=data, filetype="pdf")
            
            if doc.page_count == 0:
                raise Exception("PDF has no pages")
            
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
            raise Exception(f"PDF Error: {str(e)}")
            
    elif filename.endswith('.docx'):
        try:
            file.stream.seek(0)
            file_data = file.read()
            
            if not file_data:
                raise Exception("Word document appears to be empty")
            
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
            raise Exception(f"Word Document Error: {str(e)}")
            
    else:
        raise Exception(f"Unsupported file type: {filename}. Supported: PNG, JPG, JPEG, GIF, BMP, PDF, DOCX")

    # Truncate if too long
    if len(text) > 15000:
        text = text[:15000] + "\n\n[...Content truncated...]"
    
    # Check if OCR results are meaningful (fallback logic)
    text_words = len(text.split()) if text else 0
    is_image_file = filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    
    # Generate analysis prompt based on file type and content
    if filename.endswith(('.pdf', '.docx')):
        analysis_prompt = f"""EXTRACT ACTIONABLE TASKS from this document: '{file.filename}'

DOCUMENT CONTENT:
{text}

REQUIRED OUTPUT FORMAT:

## 📋 EXTRACTED TASKS TABLE
| Task | Due Date | Priority | Status |
|------|----------|----------|---------|
| [Extract specific tasks here] | [Any mentioned dates] | [High/Medium/Low] | [Pending] |

## 📅 CRITICAL DEADLINES
- List any time-sensitive items with specific dates
- Note any recurring deadlines or milestones

## 💡 KEY INSIGHTS
- Summarize main objectives
- Identify stakeholders mentioned  
- Note any dependencies or requirements

## ⚡ IMMEDIATE NEXT STEPS
1. [Most urgent action needed]
2. [Second priority action]
3. [Third priority action]

Focus on creating structured, scannable output that transforms document content into actionable work items."""

    elif is_image_file and text_words < 5:
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
        # Good OCR results or other file types - proceed with generic analysis
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
    
    return {
        'filename': file.filename,
        'analysis_prompt': analysis_prompt,
        'extracted_text': text,
        'file_size': len(file.read()) if hasattr(file, 'read') else 0
    }

def handle_file_upload():
    """UPDATED: Handle file upload processing for integrated chat flow"""
    try:
        files = request.files.getlist('file')  # Handle multiple files
        if not files or not files[0].filename:
            return jsonify({'error': 'No files uploaded'}), 400
        
        # Get current project from form or session
        project = request.form.get('project', PROJECTS[0])
        
        # Process all uploaded files
        file_analyses = []
        for file in files:
            if file and file.filename:
                try:
                    file_data = process_single_file(file, project)
                    file_analyses.append(file_data)
                except Exception as e:
                    # Log error but continue with other files
                    print(f"Failed to process {file.filename}: {e}")
                    file_analyses.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
        
        if not file_analyses:
            return jsonify({'error': 'No files could be processed'}), 400
        
        # Create combined analysis prompt for multiple files
        if len(file_analyses) == 1:
            # Single file
            file_data = file_analyses[0]
            if 'error' in file_data:
                return jsonify({'error': f"File processing failed: {file_data['error']}"}), 400
            
            analysis_prompt = file_data['analysis_prompt']
            user_message = f"📎 {file_data['filename']}"
            
        else:
            # Multiple files
            successful_files = [f for f in file_analyses if 'error' not in f]
            failed_files = [f for f in file_analyses if 'error' in f]
            
            if not successful_files:
                errors = [f"{f['filename']}: {f['error']}" for f in failed_files]
                return jsonify({'error': f"All files failed to process: {'; '.join(errors)}"}), 400
            
            # Combine analysis prompts
            analysis_parts = []
            analysis_parts.append(f"I've uploaded {len(successful_files)} files for analysis:")
            
            for i, file_data in enumerate(successful_files, 1):
                analysis_parts.append(f"\n=== FILE {i}: {file_data['filename']} ===")
                analysis_parts.append(file_data['extracted_text'])
            
            if failed_files:
                analysis_parts.append(f"\n=== PROCESSING ERRORS ===")
                for failed in failed_files:
                    analysis_parts.append(f"- {failed['filename']}: {failed['error']}")
            
            analysis_parts.append(f"\nPlease analyze the successfully processed files and provide insights, summaries, or actionable items from the content.")
            
            analysis_prompt = "\n".join(analysis_parts)
            filenames = [f['filename'] for f in successful_files]
            user_message = f"📎 {', '.join(filenames)}"
        
        # Generate AI analysis
        use_voices = ['SyntaxPrime']  # Default to SyntaxPrime for file analysis
        random_toggle = False
        
        try:
            retrieval_ctx = enhanced_retrieve(analysis_prompt, k=5) if is_ready() else []
            response_data = generate_response(
                analysis_prompt, use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
        except Exception as e:
            print(f"AI analysis failed: {e}")
            response_data = {"SyntaxPrime": f"Files processed successfully, but AI analysis failed: {e}"}

        # Save to database and track file uploads
        save_conversation_enhanced(project, user_message, response_data)
        
        # Track each uploaded file in database
        for file_data in file_analyses:
            if 'error' not in file_data:
                filename = file_data['filename']
                file_extension = filename.split('.')[-1].upper() if '.' in filename else 'UNKNOWN'
                content_summary = file_data['extracted_text'][:500] if file_data['extracted_text'] else "No text extracted"
                track_uploaded_file(filename, file_extension, project, content_summary)
        
        # CRITICAL FIX: Return success status for AJAX handling
        return jsonify({
            'success': True,
            'message': 'Files processed successfully',
            'project': project,
            'files_processed': len(successful_files) if len(file_analyses) > 1 else 1
        })
        
    except Exception as e:
        print(f"Upload route failed: {e}")
        current_app.logger.error(f"File upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# BACKWARD COMPATIBILITY: Keep original function as fallback
def handle_file_upload_legacy():
    """Original file upload handler for backward compatibility"""
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file uploaded", 400
        
        project = request.form.get('project', PROJECTS[0])
        
        # Process single file using new function
        file_data = process_single_file(file, project)
        
        # Generate AI analysis
        use_voices = ['SyntaxPrime']
        random_toggle = False
        
        try:
            retrieval_ctx = enhanced_retrieve(file_data['analysis_prompt'], k=5) if is_ready() else []
            response_data = generate_response(
                file_data['analysis_prompt'], use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
        except Exception as e:
            print(f"AI analysis failed: {e}")
            response_data = {"SyntaxPrime": f"File processed successfully, but AI analysis failed: {e}"}

        # Save to database
        user_message = f"[File Upload] {file.filename}"
        save_conversation_enhanced(project, user_message, response_data)
        
        # Track the uploaded file
        file_extension = file.filename.split('.')[-1].upper() if '.' in file.filename else 'UNKNOWN'
        content_summary = file_data['extracted_text'][:500] if file_data['extracted_text'] else "No text extracted"
        track_uploaded_file(file.filename, file_extension, project, content_summary)

        # Redirect back to main chat with the analysis
        return redirect(f'/?project={project}#bottom-anchor')
        
    except Exception as e:
        print(f"Legacy upload route failed: {e}")
        return f"Upload Error: {str(e)}", 500
