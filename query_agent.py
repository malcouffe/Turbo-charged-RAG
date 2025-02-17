from typing import List, Optional, Union
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from smolagents import CodeAgent, DuckDuckGoSearchTool

class EnhancedQueryAgent:
    def __init__(
        self,
        openai_api_key: str,
        max_reformulations: int = 3,
        temperature: float = 0.7,
        search_mode: str = "combined"  # "rag", "web" ou "combined"
    ):
        """
        Initialise l'agent de requête amélioré en utilisant smolagents.
        
        Args:
            openai_api_key: Clé API pour smolagents (et appels OpenAI sous-jacents)
            max_reformulations: Nombre maximum de reformulations de la requête
            temperature: Température pour la génération (si utilisée)
            search_mode: Mode de recherche ("rag", "web" ou "combined")
        """
        self.max_reformulations = max_reformulations
        self.temperature = temperature
        self.search_mode = search_mode
        self.model = ChatOpenAI(model="gpt-4o")
        self.smol_agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],         # Remplacez par les outils souhaités si besoin
            model=self.model    # Choisissez le modèle désiré
        )

    def generate_reformulations(self, original_query: str) -> List[str]:
        """
        Génère plusieurs reformulations de la requête originale via smolagents.
        
        Args:
            original_query: La requête initiale de l'utilisateur.
            
        Returns:
            Liste de requêtes reformulées.
        """
        try:
            reformulations = self.smol_agent.generate_reformulations(
                original_query, num_reformulations=self.max_reformulations
            )
            if not reformulations:
                return [original_query]
            return reformulations
        except Exception:
            return [original_query]

    def web_search(self, query: str) -> List[dict]:
        """
        Effectue une recherche web via smolagents.
        
        Args:
            query: Requête de recherche.
            
        Returns:
            Liste de résultats formatés.
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
        Formate les informations de localisation de la source à partir des métadonnées.
        
        Args:
            metadata: Métadonnées du document.
            
        Returns:
            Chaîne décrivant la localisation de la source.
        """
        source_info = []
        if "source" in metadata:
            if metadata["source"] == "web_search":
                return f"Web: {metadata.get('url', 'Unknown URL')}"
            elif metadata["source"] == "rag":
                source_info.append(f"Document: {metadata.get('source_file', 'Unknown file')}")
        if "page" in metadata:
            source_info.append(f"Page {metadata['page']}")
        if "slide_number" in metadata:
            source_info.append(f"Slide {metadata['slide_number']}")
        if "paragraph_number" in metadata:
            source_info.append(f"Paragraph {metadata['paragraph_number']}")
        if "section" in metadata:
            source_info.append(f"Section: {metadata['section']}")
        return " | ".join(source_info)

    def combine_results(
        self,
        rag_results: List[dict],
        web_results: List[dict],
        original_query: str,
        max_results: int = 5
    ) -> List[dict]:
        """
        Combine et classe les résultats RAG et web à l'aide de la méthode de ranking de smolagents.
        
        Args:
            rag_results: Résultats issus du vector store (RAG).
            web_results: Résultats issus de la recherche web.
            original_query: La requête originale.
            max_results: Nombre maximum de résultats à retourner.
            
        Returns:
            Liste combinée et classée de résultats.
        """
        combined_results = rag_results + web_results
        if not combined_results:
            return []
        try:
            ranked_results = self.smol_agent.rank_results(
                combined_results, original_query, max_results
            )
            return ranked_results
        except Exception:
            return combined_results[:max_results]

    def search_with_reformulations(
        self,
        vector_store: Optional[Chroma] = None,
        original_query: str = "",
        k: int = 3,
        unique_results: bool = True
    ) -> List[dict]:
        """
        Effectue une recherche en utilisant des reformulations de la requête sur le vector store (RAG) et/ou sur le web.
        En mode "rag", si aucun résultat n'est trouvé, un fallback vers la recherche web est utilisé.
        
        Args:
            vector_store: Optionnellement, le vector store (Chroma) pour le RAG.
            original_query: La requête initiale de l'utilisateur.
            k: Nombre de résultats par requête.
            unique_results: Si True, supprime les doublons.
            
        Returns:
            Liste de résultats de recherche avec métadonnées.
        """
        reformulations = self.generate_reformulations(original_query)
        all_rag_results = []
        all_web_results = []
        seen_contents = set()

        # S'assurer que la requête d'origine figure parmi les reformulations.
        if original_query not in reformulations:
            reformulations.insert(0, original_query)

        for query in reformulations:
            # Recherche via RAG si le vector store est fourni et que le mode l'exige.
            if vector_store and self.search_mode in ["rag", "combined"]:
                rag_results = vector_store.similarity_search(query, k=k)
                # Si aucun résultat RAG n'est obtenu en mode "rag", on effectue une recherche web.
                if not rag_results and self.search_mode == "rag":
                    web_results = self.web_search(query)
                    for result in web_results:
                        if not unique_results or result["content"] not in seen_contents:
                            result["original_query"] = original_query
                            result["metadata"]["source_location"] = self.format_source_location(result["metadata"])
                            all_web_results.append(result)
                            seen_contents.add(result["content"])
                else:
                    for doc in rag_results:
                        if not unique_results or doc.page_content not in seen_contents:
                            source_location = self.format_source_location(doc.metadata)
                            all_rag_results.append({
                                "content": doc.page_content,
                                "metadata": {**doc.metadata, "source": "rag", "source_location": source_location},
                                "query_used": query,
                                "original_query": original_query
                            })
                            seen_contents.add(doc.page_content)
            # Recherche web en complément.
            if self.search_mode in ["web", "combined"]:
                web_results = self.web_search(query)
                for result in web_results:
                    if not unique_results or result["content"] not in seen_contents:
                        result["original_query"] = original_query
                        result["metadata"]["source_location"] = self.format_source_location(result["metadata"])
                        all_web_results.append(result)
                        seen_contents.add(result["content"])

        if self.search_mode == "combined":
            return self.combine_results(all_rag_results, all_web_results, original_query, max_results=k)
        elif self.search_mode == "web":
            return all_web_results
        else:  # Mode "rag"
            return all_rag_results if all_rag_results else all_web_results

    def analyze_results(self, results: List[dict]) -> dict:
        """
        Analyse les résultats de recherche pour fournir des insights sur la requête.
        
        Args:
            results: Liste de résultats obtenus via search_with_reformulations.
            
        Returns:
            Dictionnaire résumant l'analyse (nombre total, répartition par requête et type de contenu).
        """
        analysis = {
            "total_results": len(results),
            "queries_used": set(),
            "results_by_query": {},
            "content_types": {}
        }
        for result in results:
            query = result.get("query_used", "")
            content_type = result["metadata"].get("type", "unknown")
            analysis["queries_used"].add(query)
            analysis["results_by_query"][query] = analysis["results_by_query"].get(query, 0) + 1
            analysis["content_types"][content_type] = analysis["content_types"].get(content_type, 0) + 1
        analysis["queries_used"] = list(analysis["queries_used"])
        return analysis
