# ExplainAI-Chatbot-
ExplainAI – Explainable AI Chatbot for ML/DL Engineers  The interactive  chatbot ExplainAI assists AI engineers together with learners in understanding and debugging machine learning and deep learning models  through Explainable AI (XAI) methods. Users can upload their `.py` or  `.ipynb` files to receive immediate explanation about model elements and training processes. The system supports concise and  detailed explanations through LLMs and RAG methods.  

Features 
1. Users can upload Python and Jupyter notebook files.
2. The system explains preprocessing steps along with  model architecture and training and evaluation processes
3. Users can select either brief or elaborate explanations
4. The system provides simulated runtime XAI through LIME / SHAP / Captum.
5. The system uses Retrieval-Augmented Generation (RAG) to deliver accurate context.
6. The system integrates web search capabilities through SerpAPI (optional).
7. Streamlit, LangChain  and Groq/Gemini and OpenAI tools enable the system development

Folder Structure
<pre> ###  Project Folder Structure ``` AI_UseCase/ ├── app.py ├── requirements.txt ├── config/ │ └── config.py ├── models/ │ ├── llm.py │ └── embedings.py ├── utils/ │ └── helper_functions.py ├── uploads/ │ └── [Uploaded files go here] └── docs/ └── README.md ``` </pre>
