from typing import List, Optional, Dict, Any, Tuple
import os
import uuid
from datetime import datetime

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Pour les PDF
from langchain_community.document_loaders import PyPDFLoader
# Pour les PPTX
from pptx import Presentation
# Pour les DOCX
import docx


class DocumentVectorStorePipeline:
    def __init__(
        self,
        openai_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialise la pipeline pour PDF, PPTX et DOCX uniquement.

        Args:
            openai_api_key: Clé API pour les embeddings OpenAI.
            chunk_size: Taille de chaque morceau de texte.
            chunk_overlap: Chevauchement entre les morceaux.
            persist_directory: Répertoire de persistance pour le vector store.
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        self.persist_directory = persist_directory

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Traite un fichier PDF."""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        chunks = []
        for page in pages:
            chunks.append({
                "content": page.page_content,
                "metadata": page.metadata
            })
        return chunks

    def process_pptx(self, file_path: str) -> List[Dict[str, Any]]:
        """Traite un fichier PPTX."""
        prs = Presentation(file_path)
        chunks = []
        for slide_number, slide in enumerate(prs.slides, 1):
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text += shape.text + "\n"
            if slide_text.strip():
                chunks.append({
                    "content": slide_text.strip(),
                    "metadata": {"source": file_path, "slide_number": slide_number}
                })
        return chunks

    def process_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Traite un fichier DOCX."""
        doc = docx.Document(file_path)
        chunks = []
        for para_number, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.text.strip():
                chunks.append({
                    "content": paragraph.text.strip(),
                    "metadata": {"source": file_path, "paragraph_number": para_number}
                })
        return chunks

    def process_document(self, file_path: str) -> Tuple[Chroma, List[LangchainDocument]]:
        """
        Traite un document (PDF, PPTX ou DOCX), crée des objets Document,
        les découpe en morceaux puis construit le vector store.
        
        Returns:
            Un tuple contenant le vector store et la liste des objets Document.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            chunks = self.process_pdf(file_path)
        elif file_extension == '.pptx':
            chunks = self.process_pptx(file_path)
        elif file_extension == '.docx':
            chunks = self.process_docx(file_path)
        else:
            raise ValueError(f"Type de fichier non supporté: {file_extension}")

        # Crée des objets Document LangChain à partir des chunks
        documents = [
            LangchainDocument(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in chunks
        ]
        # Découpage supplémentaire si nécessaire
        final_docs = self.text_splitter.split_documents(documents)
        # Assigner un id à chaque document si non défini
        for doc in final_docs:
            if not hasattr(doc, "id") or doc.id is None:
                doc.id = str(uuid.uuid4())
        # Construire le vector store via Chroma
        vector_store = Chroma.from_documents(
            documents=final_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        if self.persist_directory:
            vector_store.persist()  # Attention: méthode dépréciée depuis Chroma 0.4.x (les docs sont persistés automatiquement)
        return vector_store, final_docs

    def process_directory(self, directory_path: str) -> Tuple[Chroma, Dict[str, List[LangchainDocument]]]:
        """
        Traite tous les documents supportés dans un répertoire et combine les documents.

        Returns:
            Un tuple contenant le vector store combiné et un dictionnaire
            associant le chemin de chaque fichier à sa liste de documents.
        """
        all_docs: List[LangchainDocument] = []
        docs_by_file = {}
        supported_extensions = {'.pdf', '.pptx', '.docx'}

        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in supported_extensions:
                file_path = os.path.join(directory_path, filename)
                vector_store, docs = self.process_document(file_path)
                docs_by_file[file_path] = docs
                all_docs.extend(docs)

        combined_vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        if self.persist_directory:
            combined_vector_store.persist()
        return combined_vector_store, docs_by_file
