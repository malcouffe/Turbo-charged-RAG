from typing import List, Optional, Dict, Any, Union
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from markitdown import PdfProcessor
from pptx import Presentation
from docx import Document
import os
import base64
from datetime import datetime
from openai import OpenAI
from PIL import Image
import io

class DocumentVectorStorePipeline:
    def __init__(
        self,
        openai_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: Optional[str] = None,
        image_output_dir: Optional[str] = "extracted_images"
    ):
        """
        Initialize the document processing pipeline.
        
        Args:
            openai_api_key: API key for OpenAI embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist vector store (optional)
            image_output_dir: Directory to save extracted images
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        self.persist_directory = persist_directory
        self.image_output_dir = image_output_dir
        os.makedirs(image_output_dir, exist_ok=True)

    def analyze_image(self, image_path: str) -> str:
        """
        Use GPT-4 Vision to analyze an image and return a textual description.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Detailed description of the image
        """
        # Read and encode image
        with Image.open(image_path) as img:
            # Resize image if it's too large (max 2048px on longest side)
            if max(img.size) > 2048:
                ratio = 2048 / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        # Get description from GPT-4 Vision
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please provide a detailed description of this image, including any visible text, key elements, and their relationships."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content

    def process_image_for_vectorstore(self, image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image and add its analysis to metadata.
        
        Args:
            image_metadata: Dictionary containing image information
            
        Returns:
            Dict with updated metadata including image analysis
        """
        image_path = image_metadata["image_path"]
        image_analysis = self.analyze_image(image_path)
        
        return {
            **image_metadata,
            "analysis": image_analysis
        }

    def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF using markitdown.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        processor = PdfProcessor()
        images = []
        
        # Process the PDF and extract images
        pdf_content = processor.process_pdf(pdf_path)
        
        for idx, image in enumerate(pdf_content.images):
            # Generate unique filename for the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"image_{timestamp}_{idx}.png"
            image_path = os.path.join(self.image_output_dir, image_filename)
            
            # Save the image
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image.base64_data))
            
            # Create image metadata
            image_metadata = {
                "image_path": image_path,
                "page_number": image.page_number,
                "width": image.width,
                "height": image.height,
                "caption": image.caption if hasattr(image, 'caption') else None
            }
            images.append(image_metadata)
        
        return images

    def process_pptx(self, file_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process PowerPoint files, extracting text and images.
        
        Returns:
            Tuple of (list of text chunks with metadata, list of image metadata)
        """
        prs = Presentation(file_path)
        text_chunks = []
        images = []
        
        for slide_number, slide in enumerate(prs.slides, 1):
            # Extract text from slide
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + "\n"
            
            if slide_text.strip():
                text_chunks.append({
                    "content": slide_text.strip(),
                    "metadata": {
                        "source": file_path,
                        "slide_number": slide_number
                    }
                })
            
            # Extract images from slide
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"pptx_image_{timestamp}_{slide_number}.png"
                    image_path = os.path.join(self.image_output_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(shape.image.blob)
                    
                    images.append({
                        "image_path": image_path,
                        "slide_number": slide_number,
                        "width": shape.width,
                        "height": shape.height
                    })
        
        return text_chunks, images

    def process_docx(self, file_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process Word documents, extracting text and images.
        
        Returns:
            Tuple of (list of text chunks with metadata, list of image metadata)
        """
        doc = Document(file_path)
        text_chunks = []
        images = []
        
        # Extract text by paragraphs
        for para_number, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.text.strip():
                text_chunks.append({
                    "content": paragraph.text.strip(),
                    "metadata": {
                        "source": file_path,
                        "paragraph_number": para_number
                    }
                })
        
        # Extract images
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"docx_image_{timestamp}_{os.path.basename(rel.target_ref)}"
                image_path = os.path.join(self.image_output_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(rel.target_part.blob)
                
                images.append({
                    "image_path": image_path,
                    "source": file_path
                })
        
        return text_chunks, images

    def process_document(self, file_path: str) -> tuple[Chroma, List[Dict[str, Any]]]:
        """
        Process any supported document (PDF, PPTX, DOCX) and store it in the vector store.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            # For PDFs, we'll process text and images together to maintain context
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Extract and analyze images
            images = self.extract_images(file_path)
            analyzed_images = [self.process_image_for_vectorstore(img) for img in images]
            
            # Group images by page
            images_by_page = {}
            for img in analyzed_images:
                page_num = img['page_number']
                if page_num not in images_by_page:
                    images_by_page[page_num] = []
                images_by_page[page_num].append(img)
            
            # Combine text with image analysis in context
            enhanced_pages = []
            for page in pages:
                page_num = page.metadata['page']
                page_images = images_by_page.get(page_num, [])
                
                # Create combined text with image descriptions in context
                enhanced_text = page.page_content
                for img in page_images:
                    # Add image analysis after the original text
                    enhanced_text += f"\n\nImage Analysis: {img['analysis']}"
                
                enhanced_pages.append(
                    Document(
                        page_content=enhanced_text,
                        metadata={
                            **page.metadata,
                            "type": "combined",
                            "images": page_images
                        }
                    )
                )
            
            # Split enhanced pages into chunks
            chunks = self.text_splitter.split_documents(enhanced_pages)
            
        elif file_extension == '.pptx':
            text_chunks, images = self.process_pptx(file_path)
            analyzed_images = [self.process_image_for_vectorstore(img) for img in images]
            
            # Group images by slide
            images_by_slide = {}
            for img in analyzed_images:
                slide_num = img['slide_number']
                if slide_num not in images_by_slide:
                    images_by_slide[slide_num] = []
                images_by_slide[slide_num].append(img)
            
            # Combine text with image analysis for each slide
            enhanced_chunks = []
            for chunk in text_chunks:
                slide_num = chunk['metadata']['slide_number']
                slide_images = images_by_slide.get(slide_num, [])
                
                # Combine text with image descriptions
                enhanced_text = chunk['content']
                for img in slide_images:
                    enhanced_text += f"\n\nImage Analysis: {img['analysis']}"
                
                enhanced_chunks.append(
                    Document(
                        page_content=enhanced_text,
                        metadata={
                            **chunk['metadata'],
                            "type": "combined",
                            "images": slide_images
                        }
                    )
                )
            chunks = enhanced_chunks
            
        elif file_extension == '.docx':
            text_chunks, images = self.process_docx(file_path)
            analyzed_images = [self.process_image_for_vectorstore(img) for img in images]
            
            # Since Word documents don't have a clear image-paragraph mapping,
            # we'll insert image descriptions at their approximate locations
            enhanced_chunks = []
            images_per_chunk = len(analyzed_images) // len(text_chunks) + 1
            
            for i, chunk in enumerate(text_chunks):
                start_idx = i * images_per_chunk
                end_idx = start_idx + images_per_chunk
                chunk_images = analyzed_images[start_idx:end_idx]
                
                # Combine text with image descriptions
                enhanced_text = chunk['content']
                for img in chunk_images:
                    enhanced_text += f"\n\nImage Analysis: {img['analysis']}"
                
                enhanced_chunks.append(
                    Document(
                        page_content=enhanced_text,
                        metadata={
                            **chunk['metadata'],
                            "type": "combined",
                            "images": chunk_images
                        }
                    )
                )
            chunks = enhanced_chunks
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create vector store from enhanced chunks
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        if self.persist_directory:
            vector_store.persist()
            
        return vector_store, analyzed_images

    def process_directory(self, directory_path: str) -> tuple[Chroma, Dict[str, List[Dict[str, Any]]]]:
        """
        Process all supported documents in a directory.
        """
        all_chunks = []
        all_images = {}
        
        supported_extensions = {'.pdf', '.pptx', '.docx'}
        
        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in supported_extensions:
                file_path = os.path.join(directory_path, filename)
                vector_store, images = self.process_document(file_path)
                all_images[file_path] = images
                
                # Get documents from vector store
                all_chunks.extend(vector_store.get())
        
        # Create combined vector store
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        if self.persist_directory:
            vector_store.persist()
            
        return vector_store, all_images 