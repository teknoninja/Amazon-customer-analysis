import streamlit as st
from agent import SupportAgent
import logging

st.set_page_config(page_title="AmaJohnHelp AI Support", page_icon="📦", layout="centered")

st.title("📦 AmaJohnHelp AI Support Agent")
st.markdown("This AI agent uses RAG based on historical `@AmaJohnHelp` interactions to draft replies, classify intent, and decide whether to escalate to higher executives or not.")

# Initialize the agent in session state so it's only loaded once
if "agent" not in st.session_state:
    with st.spinner("Initializing AI Support Agent (loading ChromaDB embeddings)..."):
        st.session_state.agent = SupportAgent()
    st.success("Agent initialized!")

st.markdown("---")

# User input form
with st.form("query_form"):
    customer_message = st.text_area("Customer Message:", placeholder="e.g. My package was supposed to arrive yesterday but the tracking hasn't updated. Where is it?", height=100)
    submitted = st.form_submit_button("Process Message")

if submitted:
    if not customer_message.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Processing with Groq..."):
            try:
                # Process the message
                response = st.session_state.agent.process_message(customer_message)
                
                st.markdown("### Agent Output")
                
                # Display Results in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Intent", value=response.intent)
                with col2:
                    action_color = "red" if response.action == "escalate" else "green"
                    st.markdown(f"**Action:** :{action_color}[{response.action.upper()}]")
                
                st.markdown(f"**Reason:** {response.reason}")
                
                st.markdown("### Draft Reply")
                st.info(response.draft_reply)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
