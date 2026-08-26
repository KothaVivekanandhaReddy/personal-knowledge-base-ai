from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag import RAGPipeline


app = FastAPI(
    title="Personal Knowledge Base AI",
    description="RAG-based question answering system",
    version="1.0.0"
)


class QueryRequest(BaseModel):

    question: str


class Source(BaseModel):

    document_name: str
    page_number: int


class QueryResponse(BaseModel):

    answer: str
    sources: list[Source]


print("Initializing RAG pipeline...")

rag_pipeline = RAGPipeline()

print("Application ready.")


@app.get("/")
def root():

    return {
        "message": "Personal Knowledge Base AI is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/query",
    response_model=QueryResponse
)
def query_documents(
    request: QueryRequest
):

    result = rag_pipeline.query(
        request.question
    )

    return result