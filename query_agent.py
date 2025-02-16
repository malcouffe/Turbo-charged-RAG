from typing import List, Optional, Union
from openai import OpenAI
from langchain.vectorstores import Chroma
from smol_dev.agent import SmolAgent
import json

class EnhancedQueryAgent:
    def __init__(
        self,
        openai_api_key: str,
        max_reformulations: int = 3,
        temperature: float = 0.7,
        search_mode: str = "combined"
    ):
        """
        Initialize the enhanced query agent.
        
        Args:
            openai_api_key: API key for OpenAI
            max_reformulations: Maximum number of query reformulations to generate
            temperature: Temperature for query generation
            search_mode: One of "rag", "web", or "combined"
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.max_reformulations = max_reformulations
        self.temperature = temperature
        self.search_mode = search_mode
        self.smol_agent = SmolAgent(openai_api_key=openai_api_key)

    def generate_reformulations(self, original_query: str) -> List[str]:
        """
        Generate multiple reformulations of the original query.
        
        Args:
            original_query: The user's original query
            
        Returns:
            List of reformulated queries
        """
        system_prompt = """You are an expert at reformulating search queries to improve document retrieval. 
        Your task is to generate alternative versions of a query that:
        1. Capture the same intent but use different wording
        2. Include relevant synonyms and related terms
        3. Consider both specific and broader aspects of the query
        4. Account for different ways the information might be expressed in documents
        
        Return your response as a JSON array of strings, each containing a reformulated query."""

        user_prompt = f"""Original query: "{original_query}"
        
        Generate {self.max_reformulations} different versions of this query that might help find relevant information in a document collection.
        Focus on maintaining the original intent while varying the language and scope."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            response_format={ "type": "json_object" }
        )

        try:
            reformulations = json.loads(response.choices[0].message.content)
            return reformulations.get("queries", [original_query])
        except json.JSONDecodeError:
            return [original_query]

    def web_search(self, query: str) -> List[dict]:
        """
        Perform a web search using smolAgent.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        search_results = self.smol_agent.search(query)
        return [
            {
                "content": result.get("snippet", ""),
                "metadata": {
                    "type": "web_search",
                    "url": result.get("link", ""),
                    "title": result.get("title", "")
                },
                "query_used": query
            }
            for result in search_results
        ]

    def format_source_location(self, metadata: dict) -> str:
        """
        Format the source location information from metadata.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Formatted source location string
        """
        source_info = []
        
        # Get document source
        if "source" in metadata:
            if metadata["source"] == "web_search":
                return f"Web: {metadata.get('url', 'Unknown URL')}"
            elif metadata["source"] == "rag":
                source_info.append(f"Document: {metadata.get('source_file', 'Unknown file')}")
        
        # Get specific location in document
        if "page" in metadata:
            source_info.append(f"Page {metadata['page']}")
        if "slide_number" in metadata:
            source_info.append(f"Slide {metadata['slide_number']}")
        if "paragraph_number" in metadata:
            source_info.append(f"Paragraph {metadata['paragraph_number']}")
        
        # Add any section or chapter information if available
        if "section" in metadata:
            source_info.append(f"Section: {metadata['section']}")
        
        return " | ".join(source_info)

    def combine_results(
        self,
        rag_results: List[dict],
        web_results: List[dict],
        max_results: int = 5
    ) -> List[dict]:
        """
        Combine and rank RAG and web search results.
        
        Args:
            rag_results: Results from RAG search
            web_results: Results from web search
            max_results: Maximum number of results to return
            
        Returns:
            Combined and ranked results
        """
        # Use GPT to score and rank the combined results
        combined_results = rag_results + web_results
        if not combined_results:
            return []

        scoring_prompt = """Rate the relevance and quality of each search result on a scale of 0-10.
        Consider:
        1. Relevance to the original query
        2. Information completeness
        3. Source reliability (internal documents vs web)
        4. Specificity of the information
        5. Context and location of the information in the source
        
        Return a JSON object with scores for each result index."""

        results_context = "\n".join([
            f"Result {i}: [From: {result['metadata']['source_location']}]\n{result['content'][:200]}..."
            for i, result in enumerate(combined_results)
        ])

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": scoring_prompt},
                {"role": "user", "content": f"Original query: {combined_results[0]['query_used']}\n\nResults:\n{results_context}"}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )

        try:
            scores = json.loads(response.choices[0].message.content)
            scored_results = [
                (float(scores.get(str(i), 0)), result)
                for i, result in enumerate(combined_results)
            ]
            scored_results.sort(reverse=True)
            return [result for _, result in scored_results[:max_results]]
        except (json.JSONDecodeError, ValueError):
            return combined_results[:max_results]

    def search_with_reformulations(
        self,
        vector_store: Optional[Chroma] = None,
        original_query: str = "",
        k: int = 3,
        unique_results: bool = True
    ) -> List[dict]:
        """
        Search using multiple reformulated queries across RAG and/or web.
        
        Args:
            vector_store: Optional Chroma vector store for RAG
            original_query: The user's original query
            k: Number of results to return per query
            unique_results: Whether to remove duplicate results
            
        Returns:
            List of search results with metadata
        """
        reformulations = self.generate_reformulations(original_query)
        all_rag_results = []
        all_web_results = []
        seen_contents = set()

        if original_query not in reformulations:
            reformulations.insert(0, original_query)

        for query in reformulations:
            # RAG search if vector store is provided
            if vector_store and self.search_mode in ["rag", "combined"]:
                rag_results = vector_store.similarity_search(query, k=k)
                for doc in rag_results:
                    if not unique_results or doc.page_content not in seen_contents:
                        # Enhanced metadata with source location
                        source_location = self.format_source_location(doc.metadata)
                        all_rag_results.append({
                            "content": doc.page_content,
                            "metadata": {
                                **doc.metadata,
                                "source": "rag",
                                "source_location": source_location
                            },
                            "query_used": query,
                            "original_query": original_query
                        })
                        seen_contents.add(doc.page_content)

            # Web search with enhanced source tracking
            if self.search_mode in ["web", "combined"]:
                web_results = self.web_search(query)
                for result in web_results:
                    if not unique_results or result["content"] not in seen_contents:
                        result["original_query"] = original_query
                        result["metadata"]["source_location"] = self.format_source_location(result["metadata"])
                        all_web_results.append(result)
                        seen_contents.add(result["content"])

        # Combine and rank results if using both sources
        if self.search_mode == "combined":
            return self.combine_results(all_rag_results, all_web_results)
        elif self.search_mode == "web":
            return all_web_results
        else:
            return all_rag_results

    def analyze_results(self, results: List[dict]) -> dict:
        """
        Analyze the search results to provide insights about the queries.
        
        Args:
            results: List of search results from search_with_reformulations
            
        Returns:
            Dictionary containing analysis of the results
        """
        analysis = {
            "total_results": len(results),
            "queries_used": set(),
            "results_by_query": {},
            "content_types": {}
        }

        for result in results:
            query = result["query_used"]
            content_type = result["metadata"].get("type", "unknown")
            
            analysis["queries_used"].add(query)
            
            if query not in analysis["results_by_query"]:
                analysis["results_by_query"][query] = 0
            analysis["results_by_query"][query] += 1
            
            if content_type not in analysis["content_types"]:
                analysis["content_types"][content_type] = 0
            analysis["content_types"][content_type] += 1

        analysis["queries_used"] = list(analysis["queries_used"])
        
        return analysis 