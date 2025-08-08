# app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']

import streamlit as st
import os
import sys
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Import necessary LangChain components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from models.embeddings import get_embedding_model

# Add the parent directory to the system path to find other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.markdown("""
    <style>
        html, body, [class*="st-emotion"], [data-testid] {
            font-family: 'Segoe UI', 'Inter', sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f5f7fa !important;
            color: #1e1e1e;
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 20px;
            margin-top: 0;
        }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stSelectbox > label {
            color: #1e1e1e;
            font-weight: 600;
        }
        .stRadio div[role="radiogroup"] label {
            color: #1e1e1e;
            font-weight: 500;
        }
        .st-emotion-cache-1c7v05p.e1f1d6gn2 { background-color: #f9f9f9; border-radius: 10px; padding: 10px; }
        .st-emotion-cache-1c7v05p.e1f1d6gn4 { background-color: #eaf3ff; border-radius: 10px; padding: 10px; }
        .st-emotion-cache-159t3g8 { display: none; }

        .stFileUploader > label {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            background: #f0f2f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease-in-out;
            margin: 0;
            padding: 0;
        }
        .stFileUploader > label:hover {
            background: #dce3ea;
            transform: scale(1.05);
        }
        .stFileUploader > label > div {
            display: none;
        }

        .stChatInput input::placeholder {
            color: #888;
        }
        .thinking-spinner::after {
            content: "Thinking...";
            color: #555;
            font-style: italic;
        }

        .st-emotion-cache-1l0swjl.e1f1d6gn3 {
            padding-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# Import from local project files.
from models.llm import get_llm_model
import config.config as app_config
from utils.model_parser import parse_uploaded_code_file

# --- Tool Definitions ---

@tool
def web_search(query: str) -> str:
    """
    Performs a real-time web search for the given query using SerpApi.
    This tool should be used for general knowledge questions that are not
    related to the uploaded code file.
    """
    try:
        serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_api_key:
            return "Error: SERPAPI_API_KEY is not set. Please add it to your .env file."
        
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error performing web search with SerpApi: {e}"

@tool
def code_knowledge_search(query: str) -> str:
    """
    Searches the uploaded AI model's code file for relevant information.
    This tool is to be used for questions specifically about the user's code,
    such as model architecture, data preprocessing, or training parameters.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No code file has been uploaded for analysis. Please upload a file first."
    
    retriever = st.session_state.vectorstore.as_retriever()
    docs = retriever.invoke(query)
    code_context = "\n\n".join([doc.page_content for doc in docs])
    
    if not code_context:
        return "No relevant information found in the uploaded code for this query."
    
    return code_context

@tool
def perform_model_analysis(analysis_type: Optional[str] = "performance") -> str:
    """
    Performs a comparative analysis of the different models used in the uploaded code.
    Use this tool when the user asks for a comparative analysis, a summary of model performance,
    or a comparison of different models. The response is formatted as a Markdown table.
    
    Args:
        analysis_type (Optional[str]): The type of analysis to perform. Defaults to 'performance'.
        
    Returns:
        str: A Markdown-formatted string containing the comparative analysis table.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No code file has been uploaded for analysis. Please upload a file first."
    
    # In a real-world scenario, this tool would parse the code to find models and their metrics.
    # For this example, we'll simulate the data.
    model_performance = {
        "Logistic Regression": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85},
        "Random Forest": {"accuracy": 0.92, "precision": 0.91, "recall": 0.93, "f1_score": 0.92},
        "XGBoost": {"accuracy": 0.94, "precision": 0.95, "recall": 0.92, "f1_score": 0.93},
        "Decision Tree": {"accuracy": 0.88, "precision": 0.87, "recall": 0.89, "f1_score": 0.88},
        "SVM": {"accuracy": 0.90, "precision": 0.89, "recall": 0.91, "f1_score": 0.90},
        "KNN": {"accuracy": 0.83, "precision": 0.80, "recall": 0.85, "f1_score": 0.82},
    }
    
    # Format the comparative analysis as a Markdown table
    df = pd.DataFrame.from_dict(model_performance, orient='index')
    df.index.name = 'Model'
    analysis_string = "### Comparative Analysis of Model Performance\n\n"
    analysis_string += df.to_markdown(floatfmt=".2f")
    analysis_string += "\n\nThe **XGBoost** model demonstrated the highest accuracy and precision, making it the top performer in this comparison. **Random Forest** also performed very well, while **KNN** had the lowest accuracy."
    
    return analysis_string

@tool
def create_model_visualization(plot_type: str = "feature_importance") -> io.BytesIO:
    """
    Creates a data visualization (e.g., a bar chart for feature importance) and returns it as a PNG image buffer.
    This tool should be used when the user asks to "show me a plot", "visualize", or "create a chart"
    of model insights like feature importance.

    Returns:
        io.BytesIO: An in-memory buffer containing the image data.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No code file has been uploaded for analysis. Please upload a file first."
    
    if plot_type == "feature_importance":
        features = ['feature_A', 'feature_B', 'feature_C', 'feature_D', 'feature_E']
        importance = np.random.rand(len(features))
        importance = importance / importance.sum() # Normalize to sum to 1
        
        fig, ax = plt.subplots()
        ax.barh(features, importance, color='skyblue')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance Analysis')
        ax.set_xlim(0, 1)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig) # Close the figure to free up memory
        
        return buf
    
    return io.BytesIO()

@tool
def get_feature_explanation(feature_name: str) -> str:
    """
    Explains the impact of a specific feature on the model's predictions using a simulated SHAP-like value.
    This tool is used when a user asks about the importance of a single feature, e.g., "Explain the role of `feature_A`."
    
    Args:
        feature_name (str): The name of the feature to explain.
        
    Returns:
        str: A natural language explanation of the feature's impact.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No code file has been uploaded for analysis. Please upload a file first."

    explanations = {
        'feature_A': "Feature A has a significant **positive impact** on the model's output. Higher values for `feature_A` typically lead to a higher predicted value.",
        'feature_B': "The model's prediction is strongly influenced by `feature_B`. When its value increases, the model's output tends to **decrease**.",
        'feature_C': "Feature C has a moderate and complex relationship with the model's output. Its impact can vary depending on its interaction with other features.",
    }
    
    explanation = explanations.get(feature_name, f"Sorry, I don't have a specific explanation for `{feature_name}`. The model might not use it, or its impact may be minimal.")
    
    return explanation

# --- Streamlit Page Functions ---

def instructions_page():
    """Displays the instructions and setup guide for the chatbot."""
    st.title("💡 ExplainAI: Setup Guide")
    st.markdown("Welcome to **ExplainAI**, your dedicated assistant for understanding AI models. Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
    First, install the required dependencies:
    ```bash
    pip install -r requirements.txt
    pip install langchain-community google-search-results python-dotenv chromadb nbformat sentence-transformers matplotlib pandas tabulate shap
    ```
    
    ## API Key Setup
    You'll need API keys from your chosen provider. Set these as environment variables in a `.env` file in your project root.
    
    **Example `.env` file:**
    ```
    GROQ_API_KEY="your_groq_key_here"
    GOOGLE_API_KEY="your_google_gemini_key_here"
    SERPAPI_API_KEY="your_serpapi_key_here" # Your SerpApi key
    ```
    
    ## How to Use
    1. **Place your `.env` file** in the root of your `AI_UseCase` folder.
    2. **Go to the Chat page** using the sidebar.
    3. **Select your preferred LLM Provider** and **Explanation Style**.
    4. **Upload your model's code file** (.py or .ipynb) using the paperclip icon in the chatbox.
    5. **Ask your question** in the chatbox, and the bot will analyze the code or perform a web search to provide an explanation.
    6. **Ask for a comparative analysis** of the models (e.g., "Give me a comparative analysis of all models performance in the file").
    7. **Ask for specific explanations** (e.g., "Explain the role of `feature_A`") or **visualizations** (e.g., "Show me a plot of feature importance").
    
    ---
    
    Ready to start? Navigate to the **Chat** page! 
    """)

def chat_page():
    """Displays the main chat interface for the chatbot."""
    st.title("🧠 ExplainAI")
    st.markdown("Your dedicated assistant for understanding AI models. Upload your code file and ask for explanations.")
    
    llm_provider = st.session_state.get("llm_provider_selection", "Groq")
    explanation_style = st.session_state.get("explanation_style_selection", "Concise")
    
    chat_model = None
    api_key_found = False
    model_name_to_use = None
    
    try:
        if llm_provider == "Groq" and app_config.GROQ_API_KEY:
            api_key_found = True
            model_name_to_use = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
            chat_model = get_llm_model("Groq", model_name_to_use)
        elif llm_provider == "Google Gemini" and app_config.GOOGLE_API_KEY:
            api_key_found = True
            model_name_to_use = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
            chat_model = get_llm_model("Google Gemini", model_name_to_use)
        else:
            st.warning(f"{llm_provider} API Key not found in .env file.")
    except Exception as e:
        st.error(f"Error initializing model: {e}. Please check your API key and model name.")
        chat_model = None

    # THE UPDATED SYSTEM PROMPT
    system_prompt_template = (
        f"You are **ExplainAI**, a specialized XAI expert for AI engineers. "
        f"Your main task is to provide explanations about an AI model. "
        f"You have several tools available: `code_knowledge_search`, `web_search`, `perform_model_analysis`, `create_model_visualization`, and `get_feature_explanation`. "
        f"You MUST use `code_knowledge_search` for questions about the uploaded code. "
        f"You MUST use `web_search` for general knowledge questions. "
        f"When the user asks for a comparative analysis of models, model performance, or a summary of models, you MUST use the `perform_model_analysis` tool. "
        f"When the user asks for a visual representation or a chart of model insights (e.g., feature importance), you MUST use `create_model_visualization`. "
        f"When the user asks for an explanation of a specific feature's impact on predictions, you MUST use `get_feature_explanation`. "
        f"Your explanation must be {explanation_style.lower()}. "
        f"For questions about model performance, you **must first present the comparative analysis from the tool**, and then provide a dedicated section with **Actionable Suggestions** "
        f"for potential improvements or ways to enhance the model's interpretability. "
        f"The main answer should come first, before the suggestions."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], io.BytesIO):
                st.image(message["content"], caption="Generated Visualization")
            else:
                st.markdown(message["content"])

    if chat_model and api_key_found:
        tools = [web_search, code_knowledge_search, perform_model_analysis, create_model_visualization, get_feature_explanation]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(chat_model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        if prompt_input := st.chat_input(f"Chat with {llm_provider} ({model_name_to_use}). Ask your XAI question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            
            with st.chat_message("user"):
                st.markdown(prompt_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Getting XAI explanation..."):
                    try:
                        response = agent_executor.invoke({
                            "input": prompt_input,
                            "chat_history": st.session_state.messages
                        })
                        full_response = response.get("output", "")

                        if full_response:
                            st.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            st.error("The agent did not return a valid response. Please try again.")
                            st.session_state.messages.append({"role": "assistant", "content": "The agent did not return a valid response. Please try again."})

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})
    else:
        st.info("🔧 Please select an LLM provider and ensure its API key is set in your .env file to start chatting.")

    st.markdown("""
        <style>
        .stFileUploader > label {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            background: #f0f2f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease-in-out;
            margin: 0;
            padding: 0;
        }
        .stFileUploader > label:hover {
            background: #e6e8eb;
            transform: scale(1.05);
        }
        .stFileUploader > label > div {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    file_uploader_icon = "📎"
    
    uploaded_file = st.file_uploader(
        file_uploader_icon,
        type=["py", "ipynb"],
        key="code_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if "processed_code_file_name" not in st.session_state or \
           st.session_state.processed_code_file_name != uploaded_file.name:
            
            with st.spinner(f"Processing {uploaded_file.name} for RAG..."):
                code_content, error_msg = parse_uploaded_code_file(uploaded_file)
                
                if code_content:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = [Document(page_content=code_content, metadata={"source": uploaded_file.name})]
                    splits = text_splitter.split_documents(docs)
                    

                    embedding_model = get_embedding_model()
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=splits, 
                        embedding=embedding_model,
                    )
                    
                    st.session_state.processed_code_file_name = uploaded_file.name
                    st.session_state.messages.append({"role": "assistant", "content": f"Code from '{uploaded_file.name}' processed successfully! You can now ask questions about the model."})
                    st.rerun()
                elif error_msg:
                    st.error(f"Error processing code file: {error_msg}")
                    st.session_state.vectorstore = None
    else:
        if "processed_code_file_name" in st.session_state:
            del st.session_state.processed_code_file_name
        st.session_state.vectorstore = None

def main():
    """The main function to run the Streamlit app."""
    st.set_page_config(
        page_title="ExplainAI - AI Model Explainer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <style>
            html, body, [class*="st-emotion"], [data-testid] { font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
            [data-testid="stSidebar"] { background-color: #e0e0e0; color: #333; }
            [data-testid="stSidebar"] .st-emotion-cache-1lcbm9l { color: #333; }
            [data-testid="stSidebar"] .stRadio > label, [data-testid="stSidebar"] .stSelectbox > label { color: #333; }
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #333; }
            [data-testid="stSidebar"] .stSelectbox div[role="button"] { background-color: #f8f8f8; color: #333; }
            /* Chat message styling */
            .st-emotion-cache-1c7v05p.e1f1d6gn2 { background-color: #f0f0f0; border-radius: 10px; padding: 10px; }
            .st-emotion-cache-1c7v05p.e1f1d6gn4 { background-color: #e9f0ff; border-radius: 10px; padding: 10px; }
            /* Footer and chat input styling */
            .st-emotion-cache-159t3g8 { display: none; } /* Hide Streamlit's default footer */
            .st-emotion-cache-1d9g9d2 {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: white;
                z-index: 1000;
                padding: 10px;
                box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        st.title("ExplainAI Navigation")

        llm_provider = st.selectbox(
            "Select LLM Provider:",
            ["Groq", "Google Gemini"], 
            index=0,
            key="llm_provider_selection"
        )
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Chat"

        current_page_index = ["Chat", "Instructions"].index(st.session_state.current_page)
        
        page_selection = st.radio(
            "Go to:",
            ["Chat", "Instructions"], 
            index=current_page_index,
            key="main_navigation_radio"
        )
        st.session_state.current_page = page_selection
        
        st.subheader("Explanation Style")
        explanation_style = st.selectbox(
            "Choose explanation detail:",
            ["Concise", "Detailed"],
            index=0,
            key="explanation_style_selection"
        )
        
        st.divider()
        if st.session_state.current_page == "Chat": 
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.vectorstore = None
                if "processed_code_file_name" in st.session_state:
                    del st.session_state.processed_code_file_name
                st.rerun()

    if st.session_state.current_page == "Instructions":
        instructions_page()
    elif st.session_state.current_page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()