import streamlit as st
from document_processor import DocumentVectorStorePipeline
from query_agent import EnhancedQueryAgent
import os
from typing import List, Dict, Any
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
            
            # Update session state
            st.session_state.vector_store = vector_store
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
        st.title("Document Search and Analysis")
        
        # File upload section
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, PPTX, DOCX)",
            type=['pdf', 'pptx', 'docx'],
            accept_multiple_files=True
        )

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    self.process_uploaded_file(uploaded_file)

        # Display processed files
        if st.session_state.processed_files:
            st.write("**Processed Files:**")
            for file_name in st.session_state.processed_files:
                st.write(f"- {file_name}")

        # Search section
        st.header("Search Documents")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter your search query")
        with col2:
            search_mode = st.selectbox(
                "Search Mode",
                options=["rag", "web", "combined"],
                format_func=lambda x: x.capitalize()
            )

        if st.button("Search") and query:
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
                        k=3,
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
                    else:
                        st.info("No results found.")

if __name__ == "__main__":
    app = RAGApp()
    app.run() 