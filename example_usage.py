from document_processor import DocumentVectorStorePipeline
from query_agent import EnhancedQueryAgent
import os

# Initialize the pipeline and agent
pipeline = DocumentVectorStorePipeline(
    openai_api_key="your-api-key",
    persist_directory="./vector_store",
    image_output_dir="./extracted_images"
)

agent = EnhancedQueryAgent(
    openai_api_key="your-api-key",
    max_reformulations=3,
    search_mode="combined"  # Use both RAG and web search
)

# Process different types of documents
vector_store, images = pipeline.process_document("path/to/document.pdf")  # PDF
vector_store, images = pipeline.process_document("path/to/presentation.pptx")  # PowerPoint
vector_store, images = pipeline.process_document("path/to/document.docx")  # Word

# Or process all documents in a directory
vector_store, all_images = pipeline.process_directory("path/to/documents")

# Print information about extracted content
for doc_path, images in all_images.items():
    print(f"\nImages from {doc_path}:")
    for img in images:
        print(f"- Image saved to: {img['image_path']}")
        if 'slide_number' in img:
            print(f"  From slide: {img['slide_number']}")
        if 'page_number' in img:
            print(f"  From page: {img['page_number']}")

# Example search with combined RAG and web results
original_query = "How does the system handle error conditions?"
results = agent.search_with_reformulations(
    vector_store=vector_store,
    original_query=original_query,
    k=3,
    unique_results=True
)

# Print results with enhanced source information
print(f"\nResults for original query: '{original_query}'\n")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Source Location: {result['metadata']['source_location']}")
    print(f"Found using query: '{result['query_used']}'")
    
    if result['metadata'].get('source') == 'web_search':
        print(f"Title: {result['metadata'].get('title')}")
    
    print("\nContent:", result['content'][:200], "...")
    
    if result['metadata'].get('type') == 'combined' and result['metadata'].get('images'):
        print("\nAssociated images:")
        for img in result['metadata']['images']:
            print(f"- {img['image_path']}")
            if 'page_number' in img:
                print(f"  Located on page {img['page_number']}")
            elif 'slide_number' in img:
                print(f"  Located on slide {img['slide_number']}")

# Get analysis of the search results with source information
analysis = agent.analyze_results(results)
print("\nSearch Analysis:")
print(f"Total unique results: {analysis['total_results']}")
print("\nResults by source type:")
for content_type, count in analysis['content_types'].items():
    print(f"- {content_type}: {count} results")

# Print source distribution
print("\nSource distribution:")
source_locations = {}
for result in results:
    location = result['metadata']['source_location']
    source_locations[location] = source_locations.get(location, 0) + 1

for location, count in source_locations.items():
    print(f"- {location}: {count} results") 