import uuid
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Utilise une recherche sémantique (BM25) pour récupérer "
        "les parties de la documentation les plus pertinentes."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "La requête de recherche (utilisez une forme affirmative plutôt qu'une question)."
            ),
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # S'assurer que chaque document a un id
        for doc in docs:
            if not hasattr(doc, "id") or doc.id is None:
                doc.id = str(uuid.uuid4())
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("La requête doit être une chaîne de caractères.")
        docs = self.retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n{doc.page_content}" for i, doc in enumerate(docs)]
        )
