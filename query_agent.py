from smolagents import ToolCallingAgent, OpenAIServerModel
from retriever import RetrieverTool
from document_processor import DocumentVectorStorePipeline
import os

model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

pipeline = DocumentVectorStorePipeline(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    persist_directory="./vector_store",
)

file_path = "../data/2_LLM_prise_en_main.pdf"
vector_store, docs = pipeline.process_document(file_path)

# On passe ici la liste de documents récupérés dans le vector store
retriever_tool = RetrieverTool(docs)

agent = ToolCallingAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

agent_output = agent.run("Explain briefly to me how an LLM works")

print("Final output:")
print(agent_output)
