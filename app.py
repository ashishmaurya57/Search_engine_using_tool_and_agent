import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables from .env (optional if using os.environ)
load_dotenv()

# --- Set up tools with reasonable content limits ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper, return_direct=True)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper, return_direct=True)

search = DuckDuckGoSearchRun(name="Search", return_direct=True)

# --- Streamlit app setup ---
st.title("🔎 LangChain - Chat with Search")
"""
Chat with knowledge tools using LangChain + Streamlit!

This app integrates Arxiv, Wikipedia, and DuckDuckGo tools to search for real-world answers.

Try asking a factual or research-based question!
"""

# Sidebar for Groq API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input and agent response
if prompt := st.chat_input(placeholder="Ask me anything..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM and tools
    if api_key:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama3-8b-8192",
            streaming=True
        )
        tools = [search, arxiv, wiki]

        # Initialize agent
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        # Display agent response with streaming handler
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"❌ Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    else:
        st.warning("Please enter your Groq API Key in the sidebar.")
