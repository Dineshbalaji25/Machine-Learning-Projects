import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAI

# Set Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyD1CGspzhd-Cs_Ewjyo3jyPq9shSnb6j3g"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini LLM with LangChain
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)

# Function to execute generated Python code safely
def execute_code(code, df=None):
    """
    Executes the generated Python code safely with enhanced validation
    """
    if df is None:
        return {"result": None, "figure": None, "error": "No DataFrame found"}

    # Clean code input
    code = re.sub(r"```(python)?", "", code).strip("`")
    
    # Validate required columns
    required_columns = re.findall(r"df\[['\"](.+?)['\"]\]", code)
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return {
            "result": None,
            "figure": None,
            "error": f"Missing columns: {', '.join(missing_columns)}\nAvailable columns: {list(df.columns)}"
        }

    try:
        local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        exec(code, {}, local_vars)
        
        # Capture figure if exists
        fig = plt.gcf() if plt.get_fignums() else None
        plt.clf()  # Clear figure after capture
        
        return {
            "result": local_vars.get('result', 'Code executed successfully'),
            "figure": fig,
            "error": None
        }
    except Exception as e:
        return {"result": None, "figure": None, "error": f"Execution Error: {str(e)}"}

# Generate analysis code using Gemini
def generate_analysis_code(user_query):
    prompt = f"""
    You are a data scientist. Generate a Python script to analyze a pandas dataframe (df).
    Follow these rules:
    1. Use only these columns: {list(df.columns) if df is not None else []}
    2. Include visualization where relevant
    3. Use matplotlib or seaborn for plots
    4. Always sort time-based data chronologically
    5. Handle datetime conversion if needed

    Data is loaded into a pandas dataframe called `df`.

    User Question: {user_query}

    Your output should ONLY contain executable Python code.
    """
    response = llm.invoke(prompt)
    return response.strip()

# Create LangChain agent
def create_agent(df):
    tools = [
        Tool(
            name="Code Generator",
            func=lambda query: generate_analysis_code(query),
            description="Generates Python code for data analysis.",
        ),
        Tool(
            name="Code Executor",
            func=lambda code: execute_code(code, df),
            description="Executes Python code and returns output.",
        ),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

# Streamlit UI
st.title("📊 AI-Powered Data Visualization & Analysis with Gemini")

st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        st.write("### 📄 Preview of Uploaded Data")
        st.dataframe(df.head(3))

        # Basic dataset info
        with st.expander("🔍 Dataset Summary"):
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {list(df.columns)}")
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())

        user_query = st.text_area("Ask a question about your data:", 
                                placeholder="e.g., Show monthly sales trends with a line chart")

        if st.button("Analyze", type="primary"):
            if user_query and df is not None:
                # Validate query columns first
                mentioned_columns = re.findall(r'\b(\w+)\b', user_query)
                missing = [col for col in mentioned_columns if col.lower() not in map(str.lower, df.columns)]
                
                if missing:
                    st.error(f"⚠️ Dataset is missing mentioned columns: {', '.join(missing)}")
                    st.write("Available columns:", list(df.columns))
                else:
                    agent = create_agent(df)
                    try:
                        generated_code = agent.run(f"Generate analysis for: {user_query}")
                        
                        if "```python" in generated_code:
                            st.write("### 📝 Generated Code")
                            st.code(generated_code, language="python")

                            execution_result = execute_code(generated_code, df)
                            
                            if execution_result["error"]:
                                st.error(f"❌ {execution_result['error']}")
                            else:
                                st.write("### 📊 Results")
                                
                                if execution_result["figure"]:
                                    st.pyplot(execution_result["figure"])
                                
                                if execution_result["result"]:
                                    st.write(execution_result["result"])
                        else:
                            st.error("❌ Failed to generate valid code. Please rephrase your query.")
                    except Exception as e:
                        st.error(f"Agent Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question and upload a dataset")
    except Exception as e:
        st.error(f"File Error: {str(e)}")
else:
    st.info("ℹ️ Please upload a dataset to begin analysis")

# Add preset analysis buttons
st.sidebar.markdown("### Quick Analysis")
if st.sidebar.button("Monthly Sales Trends"):
    st.session_state.user_query = "Plot monthly total revenue trends for 2023 with a line chart"

if st.sidebar.button("Top Product Categories"):
    st.session_state.user_query = "Show top 5 product categories by total sales"

if st.sidebar.button("Customer Age Distribution"):
    st.session_state.user_query = "Create a histogram of customer ages with bins every 10 years"