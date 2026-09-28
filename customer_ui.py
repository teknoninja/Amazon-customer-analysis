import streamlit as st
from agent import SupportAgent
import logging

# Set up logging if not already done
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="AmaJohn Customer Support", page_icon="💬", layout="centered")

st.title("💬 AmaJohn Customer Support")
st.markdown("Welcome to AmaJohnHelp. How can we assist you today?")

# Initialize the agent in session state so it's only loaded once
if "agent" not in st.session_state:
    with st.spinner("Initializing AI Support Agent..."):
        st.session_state.agent = SupportAgent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Format the message history to give the agent context
    history_str = ""
    for msg in st.session_state.messages[-6:]:  # include up to last 6 messages
        role = "Customer" if msg["role"] == "user" else "Agent"
        history_str += f"{role}: {msg['content']}\n"
    
    if history_str:
        augmented_prompt = f"Previous Conversation:\n{history_str}\nLatest Customer Message: {prompt}"
    else:
        augmented_prompt = prompt

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process with the agent
    with st.spinner("Thinking..."):
        try:
            # We use the same SupportAgent, but pass the augmented_prompt to retain conversational context
            response = st.session_state.agent.process_message(augmented_prompt)
            reply_text = response.draft_reply
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(reply_text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
