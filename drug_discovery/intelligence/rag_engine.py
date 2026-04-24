"""
RAG Engine for Biomedical Intelligence

Combines Knowledge Graph retrieval with LLM-based reasoning using
LangChain, spaCy, and BioBERT embeddings.
"""

import logging

import spacy
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import OpenAI
    from langchain.vectorstores import FAISS
except ImportError:
    RetrievalQA = None

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine for drug discovery insights."""

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1-pubmed"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")

        self.vector_store = None
        self.qa_chain = None

        if RetrievalQA is None:
            logger.warning("LangChain not installed. RAG capabilities will be limited.")

    def embed_text(self, text: str) -> torch.Tensor:
        """Generate BioBERT embeddings for text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract biomedical entities using spaCy."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
        return entities

    def initialize_qa(self, documents: list[str]):
        """Initialize LangChain QA chain with provided documents."""
        if RetrievalQA is None:
            return

        embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-v1.1-pubmed")
        self.vector_store = FAISS.from_texts(documents, embeddings)

        # Use a placeholder for LLM if no API key provided
        llm = OpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=self.vector_store.as_retriever()
        )

    def ask(self, query: str) -> str:
        """Ask a question and get an answer using RAG."""
        if self.qa_chain:
            return self.qa_chain.run(query)
        else:
            return "RAG engine not fully initialized with LLM."

    def hybrid_reasoning(self, query: str, kg_context: list[dict[str, any]]) -> str:
        """Combine KG context with RAG for reasoning."""
        context_str = "\n".join([str(ctx) for ctx in kg_context])
        prompt = (
            f"Context from Knowledge Graph: {context_str}\n\n"
            f"Question: {query}\n\nAnswer based on context and medical knowledge:"
        )
        return self.ask(prompt)
