# AI Research Assistant 2.0

An AI research assistant that lets you search academic papers, upload a PDF and ask questions grounded in its actual content, and chat with an LLM that can pull in live web results when needed. Built with LangChain, Groq, ChromaDB, and Tavily.

A significant part of this project was systematically testing the system to find out exactly where and why it fails — the findings below are from that process, not assumptions.

## Live Demo

[AI Research Assistant 2.0 Live Demo](https://research-assistant-20-2dqhw7gtbhgnjym2zadyyq.streamlit.app/)

## What It Does

**Paper Search** — searches arXiv and Semantic Scholar by topic or title, generates a structured report from the abstract, supports follow-up Q&A with real conversational memory.

**PDF Q&A (RAG)** — upload a PDF, ask questions grounded in the document via retrieval-augmented generation with cross-encoder reranking.

**Web-Search Chatbot** — a general chatbot that can pull in live Tavily results for current-information questions, with memory folded into both the search query and the answer.

## Why This Isn't Just a Tutorial RAG Project

I ran five rounds of hand-written evaluation questions (50 total) against the PDF Q&A system — easy lookups, multi-step reasoning, adversarial traps, and unanswerable questions — then hand-graded every response and diagnosed *why* each failure happened.

### What I Found

**Retrieval is strong on tabular/numeric facts, noticeably weaker on prose-only facts.** Table-based questions (accuracy, recall, p-values) were answered correctly close to 100% of the time. Facts stated plainly in dense prose (dataset names, methodology details, related-work citations) failed at a meaningfully higher rate, confirmed across all five rounds.

**The system can't reliably combine facts across separate retrieved chunks.** When two needed numbers existed in the *same* chunk, multi-step arithmetic worked correctly. When they were scattered across different chunks, the system defaulted to "I could not find this information" — even after correctly retrieving both numbers individually in an earlier turn. This points to retrieval, not reasoning, as the bottleneck.

**It can misattribute a cited result as the paper's own finding.** In one test, it presented a competing method's result — cited in the related-work section — as if the authors of this paper produced it themselves. A distinct failure from a retrieval miss: it found the right passage but conflated whose result it was.

**Cross-encoder reranking (top-15 → rerank → top-4) fixed roughly half of the previously-failed retrieval questions**, including reconciling two different accuracy figures from two separate tables. It didn't fix the cross-chunk arithmetic or attribution issues — expected, since reranking changes what's retrieved, not what the model does with it.

**arXiv's title search fails on exact titles unless phrase-quoted.** An unquoted search for a confirmed-indexed paper returned five unrelated papers; quoting the title fixed it. A fallback chain (quoted title → broad keyword → shortened keyword) now guarantees non-empty results — but a realistically paraphrased query that previously returned nothing now returns five plausible but wrong results, with no signal of low confidence. A deliberate tradeoff, not an oversight.

## Features

* Search papers by topic or exact title across arXiv and Semantic Scholar
* Auto-generated structured report from any selected paper's abstract
* Follow-up Q&A on a paper's abstract, with real multi-turn memory
* PDF upload, chunking, embedding, and Chroma vector storage
* RAG-based PDF Q&A with cross-encoder reranking
* Page-level source citations, kept separate from conversational memory so the model can't learn to imitate the citation format
* General chatbot with optional live web search (Tavily)
* Conversational memory across all three chat surfaces

## Known Limitations (Found Through Testing)

* Weaker retrieval on prose-only facts vs. tabular data
* No reliable arithmetic/comparison across facts from separate chunks
* Occasional attribution of a cited result to the wrong authors
* Paraphrased title searches can return confident-looking but irrelevant results
* Chat history has no truncation — will eventually hit context limits on long conversations
* Web search quality depends entirely on Tavily's free tier

## Technologies Used

Python, Streamlit,
LangChain (Groq, Chroma, HuggingFace integrations),
Groq (Llama 3.1 8B Instant),
ChromaDB,
Hugging Face sentence-transformers,
Cross-encoder reranking (ms-marco-MiniLM-L-6-v2),
Tavily,
arXiv API,
Semantic Scholar API,


## How to Use

1. Open the live demo link.
2. **Paper search:** enter a topic or exact title, select a result, generate a report, or ask follow-ups.
3. **PDF Q&A:** upload a PDF, wait for processing, ask questions grounded in its content.
4. **Chatbot:** ask anything; enable web search for current-information questions.

Example questions that work well:
```
What is the main idea of this paper?
Summarize the methodology.
What was the reported accuracy of [model name]?
```

Example questions known to stress the system:
```
Questions requiring combining two numbers from different sections
Questions about details mentioned once, in prose, outside any table
```
