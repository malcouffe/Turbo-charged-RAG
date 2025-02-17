from typing import List, Optional, Dict, Any, Union
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from markitdown import MarkItDown
from pptx import Presentation
import docx  # Pour lire les fichiers DOCX
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
        Initialise la pipeline de traitement de documents.

        Args:
            openai_api_key: Clé API pour les embeddings OpenAI
            chunk_size: Taille des morceaux de texte
            chunk_overlap: Chevauchement entre les morceaux
            persist_directory: Répertoire pour persister le vector store (optionnel)
            image_output_dir: Répertoire où sauvegarder les images extraites
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
        Utilise GPT-4 Vision pour analyser une image et retourner une description textuelle.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            str: Description détaillée de l'image
        """
        # Lecture et encodage de l'image
        with Image.open(image_path) as img:
            # Redimensionner l'image si nécessaire (max 2048px sur le côté le plus long)
            if max(img.size) > 2048:
                ratio = 2048 / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        # Récupérer la description depuis GPT-4 Vision
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
        Traite une image et ajoute son analyse aux métadonnées.

        Args:
            image_metadata: Dictionnaire contenant les informations de l'image

        Returns:
            Dict avec les métadonnées mises à jour incluant l'analyse de l'image
        """
        image_path = image_metadata["image_path"]
        image_analysis = self.analyze_image(image_path)
        return {**image_metadata, "analysis": image_analysis}

    def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extrait les images d'un PDF en utilisant PyMuPDF (fitz).

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            Liste de dictionnaires contenant les données et métadonnées des images
        """
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"image_{timestamp}_{page_index+1}_{img_index}.{image_ext}"
                image_path = os.path.join(self.image_output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                image_metadata = {
                    "image_path": image_path,
                    "page_number": page_index + 1,
                    "width": base_image.get("width", None),
                    "height": base_image.get("height", None),
                    "caption": None
                }
                images.append(image_metadata)
        return images

    def process_pptx(self, file_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Traite un fichier PowerPoint, en extrayant le texte et les images.

        Returns:
            Tuple de (liste de morceaux de texte avec métadonnées, liste de métadonnées d'images)
        """
        prs = Presentation(file_path)
        text_chunks = []
        images = []
        for slide_number, slide in enumerate(prs.slides, 1):
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
        Traite un fichier Word, en extrayant le texte et les images.

        Returns:
            Tuple de (liste de morceaux de texte avec métadonnées, liste de métadonnées d'images)
        """
        doc = docx.Document(file_path)
        text_chunks = []
        images = []
        for para_number, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.text.strip():
                text_chunks.append({
                    "content": paragraph.text.strip(),
                    "metadata": {
                        "source": file_path,
                        "paragraph_number": para_number
                    }
                })
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
        Traite un document supporté (PDF, PPTX, DOCX) et le stocke dans le vector store.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        analyzed_images = []
        
        if file_extension == '.pdf':
            # Utiliser PyPDFLoader pour extraire le texte
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            # Extraction des images avec PyMuPDF
            images = self.extract_images(file_path)
            analyzed_images = [self.process_image_for_vectorstore(img) for img in images]
            # Fusion du texte et des analyses d'images
            enhanced_pages = []
            for page in pages:
                enhanced_text = page.page_content
                for img in analyzed_images:
                    enhanced_text += f"\n\nImage Analysis (page {img['page_number']}): {img['analysis']}"
                enhanced_pages.append(
                    LangchainDocument(
                        page_content=enhanced_text,
                        metadata={**page.metadata, "type": "combined", "images": analyzed_images}
                    )
                )
            chunks = self.text_splitter.split_documents(enhanced_pages)
        
        elif file_extension == '.pptx':
            text_chunks, images = self.process_pptx(file_path)
            analyzed_images = [self.process_image_for_vectorstore(img) for img in images]
            images_by_slide = {}
            for img in analyzed_images:
                images_by_slide.setdefault(img['slide_number'], []).append(img)
            enhanced_chunks = []
            for chunk in text_chunks:
                slide_num = chunk['metadata']['slide_number']
                slide_images = images_by_slide.get(slide_num, [])
                enhanced_text = chunk['content']
                for img in slide_images:
                    enhanced_text += f"\n\nImage Analysis: {img['analysis']}"
                enhanced_chunks.append(
                    LangchainDocument(
                        page_content=enhanced_text,
                        metadata={**chunk['metadata'], "type": "combined", "images": slide_images}
                    )
                )
            chunks = enhanced_chunks
        
        elif file_extension == '.docx':
            text_chunks, images = self.process_docx(file_path)
            analyzed_images = [self.process_image_for_vectorstore(img) for img in images]
            enhanced_chunks = []
            images_per_chunk = (len(analyzed_images) // len(text_chunks)) + 1 if text_chunks else 0
            for i, chunk in enumerate(text_chunks):
                start_idx = i * images_per_chunk
                end_idx = start_idx + images_per_chunk
                chunk_images = analyzed_images[start_idx:end_idx]
                enhanced_text = chunk['content']
                for img in chunk_images:
                    enhanced_text += f"\n\nImage Analysis: {img['analysis']}"
                enhanced_chunks.append(
                    LangchainDocument(
                        page_content=enhanced_text,
                        metadata={**chunk['metadata'], "type": "combined", "images": chunk_images}
                    )
                )
            chunks = enhanced_chunks
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
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
        Traite tous les documents supportés dans un répertoire.
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
                all_chunks.extend(vector_store.get())
        vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        if self.persist_directory:
            vector_store.persist()
        return vector_store, all_images
