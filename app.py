import streamlit as st

from research_assistant import (
    search_all_sources,
    extract_pdf_text_chunked,
    answer_with_rag,
    generate_paper_report,
    answer_question_about_selected_paper,
    chatbot_answer,
    chatbot_answer_with_search
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

if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []

if "pdf_display_history" not in st.session_state:
    st.session_state.pdf_display_history = []


def display_latest_first(history):
    """Render a flat [user, assistant, user, assistant...] list with the
    most recent Q&A pair on top, but each pair still shows You then Assistant."""
    pairs = [history[i:i + 2] for i in range(0, len(history), 2)]

    for pair in reversed(pairs):
        for message in pair:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        st.divider()


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

        display_latest_first(st.session_state.paper_chat_history)


elif page == "PDF Q&A":
    st.header("PDF Question Answering")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_data = extract_pdf_text_chunked(uploaded_pdf)
            st.session_state.pdf_chat_history = []       # clean history fed back into the model
            st.session_state.pdf_display_history = []    # decorated history shown to the user

        st.success("PDF processed successfully.")

    if st.session_state.pdf_data:
        question = st.text_input("Ask a question from the PDF")

        if st.button("Ask PDF"):
            if question.strip():
                with st.spinner("Answering..."):
                    result = answer_with_rag(
                        st.session_state.pdf_data,
                        question,
                        history=st.session_state.pdf_chat_history,
                    )

                # clean answer only — fed back into the model as history, never contains
                # the "Sources:" footer, so the model can't start imitating that formatting
                st.session_state.pdf_chat_history.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.pdf_chat_history.append({
                    "role": "assistant",
                    "content": result["answer"]
                })

                # decorated answer with sources — only for display, never fed back to the model
                st.session_state.pdf_display_history.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.pdf_display_history.append({
                    "role": "assistant",
                    "content": result["display"]
                })
            else:
                st.warning("Please enter a question.")

        display_latest_first(st.session_state.pdf_display_history)


elif page == "Chatbot":
    st.header("General Chatbot")

    use_search = st.checkbox("Search the web for current information")

    user_message = st.text_area("Ask anything")

    if st.button("Send"):
        if user_message.strip():
            with st.spinner("Thinking..."):
                if use_search:
                    reply = chatbot_answer_with_search(
                        user_message,
                        history=st.session_state.chat_history,
                    )
                else:
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

    display_latest_first(st.session_state.chat_history)