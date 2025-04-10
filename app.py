import streamlit as st
from document_processor import DocumentVectorStorePipeline
from query_agent import EnhancedQueryAgent
import os
from typing import List, Dict, Any, Tuple
import tempfile

class RAGApp:
    def __init__(self):
        # Initialize session state
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'all_images' not in st.session_state:
            st.session_state.all_images = {}
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()

        # Initialize pipeline and agent
        self.pipeline = DocumentVectorStorePipeline(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            persist_directory="./vector_store",
            image_output_dir="./extracted_images"
        )
        
        self.agent = EnhancedQueryAgent(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            max_reformulations=3,
            search_mode="combined"
        )

    def process_uploaded_file(self, uploaded_file) -> None:
        """Process an uploaded file and update the vector store."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        try:
            # Process the document
            vector_store, images = self.pipeline.process_document(file_path)
            
            # Update session state - append to existing vector store
            if st.session_state.vector_store is None:
                st.session_state.vector_store = vector_store
            else:
                # Get documents from the new vector store
                documents = [doc for doc in vector_store._collection.get()["documents"]]
                metadatas = vector_store._collection.get()["metadatas"]
                embeddings = vector_store._collection.get()["embeddings"]
                
                # Add new documents to existing Chroma instance
                st.session_state.vector_store._collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
            st.session_state.all_images.update({uploaded_file.name: images})
            st.session_state.processed_files.add(uploaded_file.name)
            
            st.success(f"Successfully processed {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(file_path)

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display search results in a structured format."""
        for i, result in enumerate(results, 1):
            with st.expander(f"Result {i} - From: {result['metadata']['source_location']}", expanded=i==1):
                st.write("**Query used:**", result['query_used'])
                
                if result['metadata'].get('source') == 'web_search':
                    st.write("**Source:**", f"[{result['metadata'].get('title')}]({result['metadata'].get('url')})")
                
                st.write("**Content:**")
                st.write(result['content'])
                
                if result['metadata'].get('type') == 'combined' and result['metadata'].get('images'):
                    st.write("**Associated Images:**")
                    for img in result['metadata']['images']:
                        if os.path.exists(img['image_path']):
                            st.image(img['image_path'], caption=f"From {img.get('page_number', img.get('slide_number', 'unknown location'))}")

    def display_analysis(self, analysis: Dict[str, Any], source_locations: Dict[str, int]) -> None:
        """Display search analysis in a structured format."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Results by Type:**")
            for content_type, count in analysis['content_types'].items():
                st.write(f"- {content_type}: {count} results")
        
        with col2:
            st.write("**Source Distribution:**")
            for location, count in source_locations.items():
                st.write(f"- {location}: {count} results")

    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(page_title="RAG Application", layout="wide")
        
        st.title("Document Search and Analysis")
        
        # Add tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Documents", "Search", "Settings"])
        
        with tab1:
            # File upload section
            st.header("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload your documents (PDF, PPTX, DOCX)",
                type=['pdf', 'pptx', 'docx'],
                accept_multiple_files=True
            )

            # Process uploaded files
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in st.session_state.processed_files:
                            self.process_uploaded_file(uploaded_file)

            # Display processed files
            if st.session_state.processed_files:
                st.write("**Processed Files:**")
                
                # Add document removal capability
                cols = st.columns([3, 1])
                with cols[0]:
                    file_to_remove = st.selectbox("Select document to remove:", list(st.session_state.processed_files))
                with cols[1]:
                    if st.button("Remove Document"):
                        st.session_state.processed_files.remove(file_to_remove)
                        if file_to_remove in st.session_state.all_images:
                            del st.session_state.all_images[file_to_remove]
                        st.success(f"Removed {file_to_remove}")
                        # Rebuild vector store would be needed here for complete solution
                        st.experimental_rerun()
                        
                for file_name in st.session_state.processed_files:
                    st.write(f"- {file_name}")
        
        with tab2:
            # Search section
            st.header("Search Documents")
            
            # Add search history functionality
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
                
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                query = st.text_input("Enter your search query")
            with col2:
                search_mode = st.selectbox(
                    "Search Mode",
                    options=["rag", "web", "combined"],
                    format_func=lambda x: x.capitalize()
                )
            with col3:
                k_results = st.slider("Results per query", min_value=1, max_value=10, value=3)

            search_clicked = st.button("Search")
            
            # Display search history
            if st.session_state.search_history:
                with st.expander("Search History", expanded=False):
                    for i, hist_query in enumerate(st.session_state.search_history):
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.write(f"- {hist_query}")
                        with cols[1]:
                            if st.button(f"Use Query", key=f"hist_{i}"):
                                query = hist_query
                                st.experimental_rerun()

            if search_clicked and query:
                # Save to history
                if query not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, query)
                    # Keep only last 5 queries
                    st.session_state.search_history = st.session_state.search_history[:5]
                    
                if not st.session_state.vector_store and search_mode != "web":
                    st.warning("Please upload some documents first.")
                else:
                    with st.spinner("Searching..."):
                        # Update agent's search mode
                        self.agent.search_mode = search_mode
                        
                        # Perform search
                        results = self.agent.search_with_reformulations(
                            vector_store=st.session_state.vector_store,
                            original_query=query,
                            k=k_results,
                            unique_results=True
                        )
                        
                        if results:
                            # Display results
                            st.subheader("Search Results")
                            self.display_results(results)
                            
                            # Display analysis
                            st.subheader("Search Analysis")
                            analysis = self.agent.analyze_results(results)
                            source_locations = {}
                            for result in results:
                                location = result['metadata']['source_location']
                                source_locations[location] = source_locations.get(location, 0) + 1
                            
                            self.display_analysis(analysis, source_locations)
                            
                            # Add export functionality
                            if st.button("Export Results"):
                                export_data = "\n\n".join([
                                    f"Result {i+1} - From: {result['metadata']['source_location']}\n"
                                    f"Query: {result['query_used']}\n"
                                    f"Content: {result['content']}"
                                    for i, result in enumerate(results)
                                ])
                                st.download_button(
                                    "Download Results as Text",
                                    export_data,
                                    file_name="search_results.txt",
                                    mime="text/plain"
                                )
                        else:
                            st.info("No results found.")
        
        with tab3:
            # Settings section
            st.header("Application Settings")
            
            st.subheader("Document Processing")
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
            
            st.subheader("Search Settings")
            max_reformulations = st.slider("Max Query Reformulations", min_value=1, max_value=5, value=3)
            
            if st.button("Apply Settings"):
                # Update pipeline settings
                self.pipeline.text_splitter.chunk_size = chunk_size
                self.pipeline.text_splitter.chunk_overlap = chunk_overlap
                
                # Update agent settings
                self.agent.max_reformulations = max_reformulations
                
                st.success("Settings applied successfully!")
                
            # Add a document cleaning option
            st.subheader("Maintenance")
            if st.button("Clear All Documents"):
                st.session_state.vector_store = None
                st.session_state.all_images = {}
                st.session_state.processed_files = set()
                st.success("All documents have been removed.")
                st.experimental_rerun()

if __name__ == "__main__":
    app = RAGApp()
    app.run()