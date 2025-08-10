# app.py

import streamlit as st
from VidyaSetu import VidyaSetuTutor  # Imports your main class

# --- Page Configuration ---
st.set_page_config(page_title="VidyaSetu AI Bridge of knowledge", page_icon="🤖")

# --- Page Title and Description ---
st.title("🤖 VidyaSetu AI Bridge of knowledge")
st.markdown(
    """
Welcome to your personal AI tutor! Ask questions about your documents(THEMES IN WORLD HISTORY, Class XI) and get answers directly from the source material.
"""
)

# Sidebar for Reference Documents
st.sidebar.header("Reference Documents")

# Chat Interface
st.subheader("Chat with VidyaSetu")

# --- Application Logic ---

# Initialize the tutor in Streamlit's session state.
if "tutor" not in st.session_state:
    with st.spinner("📚 Preparing the tutor... This may take a moment."):
        st.session_state.tutor = VidyaSetuTutor()

# Initialize the chat message history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# *** FIX 1: Initialize the response ID for conversation state ***
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Display past chat messages from history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Handling ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the AI's response using the .ask() method from your tutor.
    with st.spinner("🤖 Thinking..."):
        # *** FIX 2: Pass the previous response ID to the ask method ***
        response, formatted_docs = st.session_state.tutor.ask(
            prompt, st.session_state.previous_response_id
        )

    # Display the AI's response.
    with st.chat_message("assistant"):
        st.markdown(response.output_text)

    # Display the retrieved reference documents
    st.sidebar.markdown("### Retrieved Documents:")
    st.sidebar.markdown(formatted_docs)

    # Add the AI's response to our history.
    st.session_state.messages.append(
        {"role": "assistant", "content": response.output_text}
    )

    # *** FIX 3: Update the response ID for the next turn ***
    st.session_state.previous_response_id = response.id
