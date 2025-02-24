import os
import re
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Gemini model and LangChain components
if "model" not in st.session_state:
    st.session_state.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    template = """You are a Python coding assistant. 
    Generate Python code based on the user request.
    Your response must ONLY contain the code block inside triple backticks.
    Do not include any explanations, only code.

    User Prompt: {prompt}"""
    
    st.session_state.prompt_template = PromptTemplate(template=template, input_variables=["prompt"])
    st.session_state.chain = LLMChain(llm=st.session_state.model, prompt=st.session_state.prompt_template)

def extract_code_from_response(response: str) -> str:
    """Extract code block from model response or return raw text if no code block is found."""
    code_pattern = r"```python\n(.*?)\n```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    return matches[0].strip() if matches else response.strip()

def execute_python_code_safely(code: str) -> str:
    """Execute Python code in a subprocess securely with a timeout."""
    try:
        result = subprocess.run(
            ['python', '-c', code],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out."
    
    except Exception as e:
        return f"Error: {str(e)}"

def code_agent(user_prompt: str) -> tuple:
    """Handle code generation and execution workflow."""
    response = st.session_state.chain.run(user_prompt)
    code = extract_code_from_response(response)
    execution_result = execute_python_code_safely(code) if code else "No valid code generated."
    return code, execution_result

# Streamlit UI
st.title("🤖 AI-Powered Code Generator & Executor")
st.caption("A coding assistant that generates and executes Python code using Gemini.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your coding request:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and execute code
    with st.spinner("Generating and executing code..."):
        code, result = code_agent(prompt)

    # Display generated code
    with st.chat_message("assistant"):
        st.markdown(f"**Generated Code:**\n```python\n{code}\n```")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Generated Code:\n```python\n{code}\n```"
        })

    # Display execution result
    with st.chat_message("assistant"):
        st.markdown(f"**Execution Result:**\n{result}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Execution Result:\n{result}"
        })
