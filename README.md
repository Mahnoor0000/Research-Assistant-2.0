# AI Research Assistant 2.0

AI Research Assistant 2.0 is an improved version of my earlier AI Research Assistant project. This version is designed to make research easier by allowing users to upload PDF documents and ask questions directly from the uploaded content using a Retrieval-Augmented Generation approach.

The main goal of this project is to help students, researchers, and professionals quickly understand long research papers, reports, notes, and academic documents without manually reading the entire PDF.

## Live Demo

[AI Research Assistant 2.0 Live Demo](https://research-assistant-20-2dqhw7gtbhgnjym2zadyyq.streamlit.app/)

## About the Project

Reading long research papers and academic documents can be time-consuming. AI Research Assistant 2.0 solves this problem by allowing users to interact with a PDF through natural language questions.

The app extracts text from an uploaded PDF, breaks the content into smaller chunks, stores those chunks in a vector database, and retrieves the most relevant information when the user asks a question. The retrieved context is then passed to a language model to generate a meaningful and document-based answer.

This makes the assistant more useful for research because the answers are based on the uploaded document rather than only relying on general AI knowledge.

## Key Features

- Upload and process PDF documents
- Extract readable text from research papers and reports
- Split long documents into smaller searchable chunks
- Generate embeddings for document chunks
- Store document embeddings in a vector database
- Ask questions from the uploaded PDF
- Retrieve the most relevant document content
- Generate context-aware answers using an LLM
- Simple and user-friendly Streamlit interface
- Improved RAG-based workflow compared to the earlier version

## Project Approach

This project follows a Retrieval-Augmented Generation pipeline.

First, the user uploads a PDF document through the Streamlit interface. The app reads the document and extracts its text. Since long documents cannot be passed directly to the language model, the extracted text is divided into smaller chunks.

After chunking, embeddings are generated for each chunk. These embeddings convert the meaning of the text into numerical form. The chunks and their embeddings are then stored inside a Chroma vector database.

When the user asks a question, the question is also converted into an embedding. The vector database compares the question embedding with the stored document embeddings and retrieves the most relevant chunks.

These retrieved chunks are used as context for the language model. The model then generates an answer based on the relevant document content. This improves the quality of the response because the answer is grounded in the uploaded PDF.

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Groq LLM
- Hugging Face embeddings
- PDF text extraction
- Retrieval-Augmented Generation

## How to Use the App

1. Open the live demo link.
2. Upload a PDF document.
3. Wait for the app to process the document.
4. Ask a question related to the uploaded PDF.
5. The app retrieves relevant content and generates an answer.

Example questions:

```text
What is the main idea of this paper?
Summarize the methodology.
What are the key findings?
Explain the conclusion in simple words.
What problem does this research solve?

```

# Improvements Over Earlier Version

AI Research Assistant 2.0 improves my earlier AI Research Assistant project by adding a more complete Retrieval-Augmented Generation workflow.


## Main improvements include:

Better PDF-based question answering
Improved document chunking
Vector search using ChromaDB
Cleaner Streamlit interface
More accurate answers using retrieved document context
Better support for research papers and academic documents
