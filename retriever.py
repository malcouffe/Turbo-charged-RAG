import uuid
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"
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
