import os
import tempfile
import requests
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load API keys from .env file
load_dotenv()


# Groq model is initialized once and reused for all LLM tasks
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# Prompt used for PDF RAG question answering
rag_prompt = PromptTemplate.from_template("""
You are a research assistant.

Answer the question using only the PDF context below.
If the answer is not in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
""")


# Prompt used to generate a report from a paper abstract
paper_report_prompt = PromptTemplate.from_template("""
Create a simple research paper report.

Title: {title}
Authors: {authors}
Year: {year}

Abstract:
{abstract}

Write:
1. Summary
2. Main contribution
3. Methodology
4. Strengths
5. Limitations

Use only the given abstract.
Do not make up details.
""")


# Prompt used to answer questions about a searched paper
paper_qa_prompt = PromptTemplate.from_template("""
Answer using only the paper abstract.

Title:
{title}

Abstract:
{abstract}

Question:
{question}

Answer:
""")


# Prompt used for the general chatbot
chat_prompt = PromptTemplate.from_template("""
You are a helpful research assistant.

Question:
{question}

Answer clearly and simply:
""")


def search_semantic_scholar(query, max_results=5):
    # Semantic Scholar returns paper data in JSON format
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,url"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        papers = []

        # Convert API response into a simple list of paper dictionaries
        for paper in data.get("data", []):
            papers.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract") or "",
                "authors": [
                    author.get("name", "")
                    for author in paper.get("authors", [])
                ],
                "year": paper.get("year", "Unknown"),
                "url": paper.get("url", ""),
                "source": "Semantic Scholar"
            })

        return papers

    except Exception:
        # Return empty list if API fails, so the app does not crash
        return []


def search_arxiv(query, max_results=5):
    # arXiv returns results in XML format
    url = "http://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        # Parse XML response
        root = ET.fromstring(response.content)

        namespace = "{http://www.w3.org/2005/Atom}"
        papers = []

        # Extract useful information from every arXiv entry
        for entry in root.findall(f"{namespace}entry"):
            title = entry.find(f"{namespace}title")
            abstract = entry.find(f"{namespace}summary")
            link = entry.find(f"{namespace}id")
            published = entry.find(f"{namespace}published")

            authors = []

            for author in entry.findall(f"{namespace}author"):
                name = author.find(f"{namespace}name")
                if name is not None:
                    authors.append(name.text)

            papers.append({
                "title": title.text.strip() if title is not None else "",
                "abstract": abstract.text.strip() if abstract is not None else "",
                "authors": authors,
                "year": published.text[:4] if published is not None else "Unknown",
                "url": link.text if link is not None else "",
                "source": "arXiv"
            })

        return papers

    except Exception:
        # Return empty list if arXiv request or XML parsing fails
        return []


def search_all_sources(query, max_results=5):
    # Search both Semantic Scholar and arXiv
    papers = search_semantic_scholar(query, max_results)
    papers += search_arxiv(query, max_results)

    unique_papers = []
    seen_titles = set()

    # Remove duplicate papers using title
    for paper in papers:
        title = paper["title"].lower()

        if title not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title)

    return unique_papers[:max_results]


def extract_pdf_text_chunked(uploaded_pdf):
    # Streamlit uploaded file is in memory, so save it temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_pdf.getvalue())
    temp_file.close()

    # Load PDF pages using LangChain PDF loader
    loader = PyPDFLoader(temp_file.name)
    pages = loader.load()

    # Convert page numbers from 0-based to 1-based
    for page in pages:
        page.metadata["page"] = page.metadata.get("page", 0) + 1

    # Split PDF text into smaller chunks for retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    # Create embeddings for semantic search
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store chunks and embeddings in Chroma vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Remove temporary PDF file after processing
    os.remove(temp_file.name)

    return {
        "vectorstore": vectorstore
    }


def answer_with_rag(pdf_data, question):
    # Convert vector database into retriever
    retriever = pdf_data["vectorstore"].as_retriever(
        search_kwargs={"k": 4}
    )

    # Retrieve top relevant chunks for the user question
    docs = retriever.invoke(question)

    context = ""

    # Combine retrieved chunks into one context string
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        context += f"Page {page}:\n{doc.page_content}\n\n"

    # Create RAG chain: prompt -> model -> text output
    chain = rag_prompt | model | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # Collect page numbers for citation/source display
    pages = [
        doc.metadata.get("page", "Unknown")
        for doc in docs
    ]

    return answer + f"\n\nSources: pages {pages}"


def generate_paper_report(paper):
    # Generate a simple report using only paper metadata and abstract
    chain = paper_report_prompt | model | StrOutputParser()

    return chain.invoke({
        "title": paper.get("title", ""),
        "authors": ", ".join(paper.get("authors", [])),
        "year": paper.get("year", "Unknown"),
        "abstract": paper.get("abstract", "")
    })


def answer_question_about_selected_paper(paper, question, history=None):
    # Answer questions using only the selected paper abstract
    chain = paper_qa_prompt | model | StrOutputParser()

    return chain.invoke({
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "question": question
    })


def chatbot_answer(question, history=None):
    # General chatbot response using the same Groq model
    chain = chat_prompt | model | StrOutputParser()

    return chain.invoke({
        "question": question
    })