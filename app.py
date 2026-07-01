import streamlit as st

from research_assistant import (
    search_all_sources,
    extract_pdf_text_chunked,
    answer_with_rag,
    generate_paper_report,
    answer_question_about_selected_paper,
    chatbot_answer,
)


st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
)

st.title("AI Research Assistant")
st.write("Search papers, upload PDFs, and ask research questions.")


if "papers" not in st.session_state:
    st.session_state.papers = []

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "paper_chat_history" not in st.session_state:
    st.session_state.paper_chat_history = []


page = st.sidebar.radio(
    "Menu",
    [
        "Search Papers",
        "PDF Q&A",
        "Chatbot",
    ],
)


if page == "Search Papers":
    st.header("Search Papers")

    topic = st.text_input("Enter research topic")

    if st.button("Search"):
        if topic.strip():
            with st.spinner("Searching papers..."):
                st.session_state.papers = search_all_sources(topic)
        else:
            st.warning("Please enter a topic.")

    if st.session_state.papers:
        paper_titles = [paper["title"] for paper in st.session_state.papers]

        selected_title = st.selectbox("Select a paper", paper_titles)
        paper = st.session_state.papers[paper_titles.index(selected_title)]

        st.subheader(paper.get("title", "Untitled"))
        st.write("Authors:", ", ".join(paper.get("authors", [])) or "Unknown")
        st.write("Year:", paper.get("year", "Unknown"))
        st.write("Source:", paper.get("source", "Unknown"))

        if paper.get("url"):
            st.write("URL:", paper["url"])

        with st.expander("Abstract"):
            st.write(paper.get("abstract", "No abstract available."))

        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                report = generate_paper_report(paper)

            st.markdown(report)

        st.subheader("Ask About This Paper")

        question = st.text_input("Question about selected paper")

        if st.button("Ask Paper"):
            if question.strip():
                with st.spinner("Answering..."):
                    answer = answer_question_about_selected_paper(
                        paper,
                        question,
                        history=st.session_state.paper_chat_history,
                    )

                st.session_state.paper_chat_history.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.paper_chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.warning("Please enter a question.")

        for message in st.session_state.paper_chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")


elif page == "PDF Q&A":
    st.header("PDF Question Answering")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_data = extract_pdf_text_chunked(uploaded_pdf)

        st.success("PDF processed successfully.")

    if st.session_state.pdf_data:
        question = st.text_input("Ask a question from the PDF")

        if st.button("Ask PDF"):
            if question.strip():
                with st.spinner("Answering..."):
                    answer = answer_with_rag(
                        st.session_state.pdf_data,
                        question
                    )

                st.markdown(answer)
            else:
                st.warning("Please enter a question.")


elif page == "Chatbot":
    st.header("General Chatbot")

    user_message = st.text_area("Ask anything")

    if st.button("Send"):
        if user_message.strip():
            with st.spinner("Thinking..."):
                reply = chatbot_answer(
                    user_message,
                    history=st.session_state.chat_history,
                )

            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reply
            })
        else:
            st.warning("Please enter a message.")

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
