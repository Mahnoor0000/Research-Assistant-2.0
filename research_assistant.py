import os
import tempfile
import requests
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder



# Load API keys from .env file
load_dotenv()


# Groq model is initialized once and reused for all LLM tasks
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# use tavily to get search data
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in .env — check your .env file exists and is loaded")

tavily_client = TavilyClient(api_key=tavily_api_key)



reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



# make a prompt template for the RAG retrieval-augmented generation
rag_prompt = PromptTemplate.from_template("""
You are a research assistant.

Answer the question using only the PDF context below.
If the answer is not in the context, say:
"I could not find this information in the document."

Do not include page numbers, citations, or a "Sources" line in your answer — 
that will be added separately.

                                          
Context:
{context}

Question:
{question}

Answer:
""")


# web search prompt template
web_search_prompt = PromptTemplate.from_template("""
You are a helpful research assistant with access to current web search results.

Web search results:
{search_results}

Previous conversation:
{history}

Question:
{question}

Answer using the search results above. Cite which source you used if relevant.
If the search results don't actually answer the question, say so honestly 
instead of guessing or using outdated knowledge.
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

Previous conversation:
{history}

Question:
{question}

Answer:
""")


# Prompt used for the general chatbot
chat_prompt = PromptTemplate.from_template("""
You are a helpful research assistant.

Previous conversation:
{history}

New question:
{question}

Answer clearly and simply, using the conversation above for context if relevant:
""")


def format_history(history):
    if not history:
        return "No previous conversation."

    history_text = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    return history_text





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
    url = "http://export.arxiv.org/api/query"
    namespace = "{http://www.w3.org/2005/Atom}"

    def run_query(search_query):
        params = {"search_query": search_query, "start": 0, "max_results": max_results}
        try:
            response = requests.get(url, params=params, timeout=10)
            root = ET.fromstring(response.content)
            papers = []
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
            return []

    # 1. try exact title phrase
    results = run_query(f'ti:"{query}"')

    # 2. fall back to broad keyword match on the full query
    if not results:
        results = run_query(f"all:{query}")

    # 3. still nothing — shrink to the last-resort case: just the first few words
    if not results:
        fallback_terms = query.split()[:4]
        results = run_query(f"all:{' '.join(fallback_terms)}")

    return results


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


def retrieve_with_rerank(vectorstore, question, initial_k=15, final_k=4):
    # cast a wider net than before
    retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})
    candidates = retriever.invoke(question)

    # score each candidate chunk against the question
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    # sort candidates by rerank score, keep the top final_k
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:final_k]]

    return top_docs

    

def search_web(query, max_results=5):
    try:
        response = tavily_client.search(query=query, max_results=max_results)
        return response.get("results", [])
    except Exception:
        # If Tavily fails (rate limit, bad key, network), return empty
        # so the caller can fall back gracefully instead of crashing.
        return []
    



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



def answer_with_rag(pdf_data, question, history=None):
    docs = retrieve_with_rerank(pdf_data["vectorstore"], question)

    context = ""
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        context += f"Page {page}:\n{doc.page_content}\n\n"

    chain = rag_prompt | model | StrOutputParser()

    raw_answer = chain.invoke({
        "history": format_history(history),
        "context": context,
        "question": question
    })

    pages = [doc.metadata.get("page", "Unknown") for doc in docs]

    return {
        "answer": raw_answer,
        "display": raw_answer + f"\n\nSources: pages {pages}"
    }



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
    # Answer using the paper abstract PLUS prior conversation turns
    chain = paper_qa_prompt | model | StrOutputParser()

    return chain.invoke({
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "history": format_history(history),
        "question": question
    })


def chatbot_answer(question, history=None):
    # General chatbot response, now actually using conversation history
    chain = chat_prompt | model | StrOutputParser()

    return chain.invoke({
        "history": format_history(history),
        "question": question
    })


def chatbot_answer_with_search(question, history=None):
    history_text = format_history(history)

    # crude but effective: fold recent history into the search query itself TO maintain context
  
    search_query = f"{history_text} {question}".strip()

    results = search_web(search_query)

    if not results:
        search_text = "No search results available."
    else:
        search_text = ""
        for r in results:
            title = r.get("title", "Untitled")
            content = r.get("content", "")
            url = r.get("url", "")
            search_text += f"- {title}: {content} (Source: {url})\n"

    chain = web_search_prompt | model | StrOutputParser()

    return chain.invoke({
        "search_results": search_text,
        "history": history_text,
        "question": question
    })