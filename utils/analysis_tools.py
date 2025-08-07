
import pandas as pd
import numpy as np
import re
import io
import streamlit as st

from typing import Optional
from langchain_core.documents import Document
from langchain_core.tools import tool

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

        /* Chat input placeholder and thinking feedback */
        .stChatInput input::placeholder {
            color: #888;
        }
        .thinking-spinner::after {
            content: "Thinking...";
            color: #555;
            font-style: italic;
        }

        /* Sidebar navigation spacing */
        .st-emotion-cache-1l0swjl.e1f1d6gn3 {
            padding-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

@tool
def perform_model_analysis(analysis_type: Optional[str] = "performance") -> str:
    """
    Analyzes the uploaded code file and extracts model types and accuracy scores.
    Returns a Markdown table and actionable suggestions based on extracted insights.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No code file has been uploaded for analysis. Please upload a file first."

    retriever = st.session_state.vectorstore.as_retriever()
    docs = retriever.invoke("model accuracy or performance metrics")

    code_context = "\n\n".join([doc.page_content for doc in docs])

    model_blocks = re.findall(
        r"([\w\s]*?(Classifier|Regressor|Model))\s*(?:=|:)?.*?(accuracy|acc)\s*[:=]\s*([\d\.]+)",
        code_context,
        re.IGNORECASE
    )

    if not model_blocks:
        return "No model performance metrics found in the uploaded code."

    records = []
    for block in model_blocks:
        model_name = block[0].strip()
        metric = float(block[3])
        records.append((model_name, metric))

    df = pd.DataFrame(records, columns=["Model", "Accuracy"])
    df.sort_values("Accuracy", ascending=False, inplace=True)

    markdown = "### Extracted Model Performance:\n\n"
    markdown += df.to_markdown(index=False)

    best_model = df.iloc[0]["Model"]
    markdown += f"\n\n🧠 **Best Performing Model:** `{best_model}`"

    markdown += "\n\n---\n### Actionable Suggestions to Improve Accuracy:\n"
    markdown += "- **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV.\n"
    markdown += f"- **Advanced Models**: Try replacing `{best_model}` with models like CatBoost or LightGBM.\n"
    markdown += "- **Cross-validation**: Use StratifiedKFold to reduce overfitting risk.\n"
    markdown += "- **Feature Selection**: Use SHAP or permutation importance to remove irrelevant features.\n"
    markdown += "- **Handling Imbalance**: Apply SMOTE or adjust class weights if data is imbalanced.\n"

    return markdown


@tool
def summarize_uploaded_notebook() -> str:
    """
    Generates a high-level summary of the uploaded notebook/code file.
    Looks for model types, data sources, training stages, and metrics.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Please upload a notebook or code file for summarization."

    retriever = st.session_state.vectorstore.as_retriever()
    docs = retriever.invoke("summarize the pipeline or purpose of this notebook")

    content = "\n\n".join([doc.page_content for doc in docs])

    if not content.strip():
        return "Could not extract summary content from the uploaded file."

    summary = "### 📄 Notebook Summary\n\n"
    if "train_test_split" in content:
        summary += "- The notebook performs a typical ML pipeline with train/test splitting.\n"
    if "XGBClassifier" in content:
        summary += "- Uses **XGBoost** for classification.\n"
    if "accuracy" in content or "score" in content:
        summary += "- Tracks performance using accuracy or score metrics.\n"
    if "sns" in content or "plt" in content:
        summary += "- Includes data visualizations using Matplotlib/Seaborn.\n"
    if "model.fit" in content:
        summary += "- Trains the model explicitly using `model.fit`.\n"

    summary += "\n\n🧠 This notebook is structured for a typical supervised ML task. You can ask specific questions like:\n"
    summary += "- What algorithm is used?\n- How is data preprocessed?\n- What are the evaluation metrics?"

    return summary


@tool
def safe_agent_executor(agent_executor, prompt_input: str) -> str:
    """
    Safely invokes an agent executor and returns the output string.
    Handles NoneType errors and adds fallback messaging.
    """
    try:
        response = agent_executor.invoke({
            "input": prompt_input,
            "chat_history": st.session_state.messages
        })

        if response and isinstance(response, dict):
            full_response = response.get("output", "")
        else:
            full_response = "The agent did not return a valid response. Please try again."

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            return full_response
        else:
            error_msg = "Empty response received. Try rephrasing or re-uploading the file."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return error_msg

    except Exception as e:
        error_msg = f"An error occurred: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return error_msg
