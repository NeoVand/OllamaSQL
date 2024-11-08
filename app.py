import streamlit as st
import pandas as pd
import requests
import ollama
import duckdb
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from http import HTTPStatus

# Constants
OLLAMA_API_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 10

@dataclass
class AIResponse:
    """Data class to structure AI agent responses"""
    sql_query: Optional[str]
    analysis: str
    sql_results: Optional[Union[List[Dict], Dict]]

class DataFrameManager:
    """Manages DataFrame operations and state"""
    @staticmethod
    def initialize_session_state() -> None:
        if 'uploaded_files' not in st.session_state:
            st.session_state['uploaded_files'] = {}
        if 'selected_df' not in st.session_state:
            st.session_state['selected_df'] = None
        if 'selected_df_file_name' not in st.session_state:
            st.session_state['selected_df_file_name'] = None

class SQLExecutor:
    """Handles SQL query execution"""
    @staticmethod
    def execute_query(query: str) -> str:
        try:
            if st.session_state['selected_df'] is None:
                return json.dumps({"error": "No dataframe selected"})
            
            with duckdb.connect() as conn:
                conn.register('selected_df', st.session_state['selected_df'])
                result_df = conn.execute(query).df()
                return json.dumps(result_df.to_dict(orient='records'))
        except Exception as e:
            return json.dumps({"error": str(e)})

class OllamaService:
    """Handles interactions with Ollama API"""
    @staticmethod
    async def get_ai_response(
        query: str,
        model_name: str,
        temperature: float
    ) -> AIResponse:
        client = ollama.AsyncClient()
        
        columns_info = ', '.join(f"'{col}'" for col in st.session_state['selected_df'].columns)
        system_prompt = f"""You are a data analysis assistant. You have access to a table 'selected_df' 
        with the following columns: {columns_info}. Generate and execute SQL queries to answer user 
        questions about this data. Always return both the SQL query and its results."""
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]
        
        # Get SQL query from model
        sql_response = await client.chat(
            model=model_name,
            messages=messages,
            options={"temperature": temperature},
            format='json',
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'execute_sql_query',
                    'description': 'Execute SQL query against the selected dataframe',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'SQL query to execute'
                            }
                        },
                        'required': ['query']
                    }
                }
            }]
        )
        
        messages.append(sql_response['message'])
        sql_results = None

        # Execute SQL query if tool call present
        if tool_calls := sql_response['message'].get('tool_calls'):
            for tool in tool_calls:
                if tool['function']['name'] == 'execute_sql_query':
                    query_args = tool['function']['arguments']
                    query_response = SQLExecutor.execute_query(query_args['query'])
                    sql_results = json.loads(query_response)
                    messages.append({
                        'role': 'tool',
                        'content': query_response,
                        'name': tool['function']['name']
                    })
        
        # Get final analysis
        final_response = await client.chat(
            model=model_name,
            messages=messages,
        )
        
        extracted_query = (
            tool_calls[0]['function']['arguments'].get('query')
            if tool_calls else None
        )
        
        return AIResponse(
            sql_query=extracted_query,
            analysis=final_response['message']['content'],
            sql_results=sql_results
        )

    @staticmethod
    def get_available_models() -> List[str]:
        try:
            response = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags")
            if response.status_code == HTTPStatus.OK:
                return [model['name'] for model in response.json()['models']]
            return []
        except requests.exceptions.RequestException:
            return []

    @staticmethod
    def is_server_running() -> bool:
        try:
            requests.get(
                f"{OLLAMA_API_BASE_URL}/api/tags",
                timeout=OLLAMA_TIMEOUT_SECONDS
            )
            return True
        except requests.exceptions.RequestException:
            return False

class StreamlitUI:
    """Handles Streamlit UI components and layout"""
    @staticmethod
    def render_sidebar() -> tuple[Optional[str], float, Optional[str]]:
        with st.sidebar:
            st.title('🦙 OllamaSQL')
            
            model_name, temperature = StreamlitUI._render_model_settings()
            user_query = StreamlitUI._render_file_upload_and_query()
            
            return model_name, temperature, user_query

    @staticmethod
    def _render_model_settings() -> tuple[Optional[str], float]:
        with st.expander("🛠️ Model Settings", expanded=False):
            model_name = None
            if OllamaService.is_server_running():
                available_models = OllamaService.get_available_models()
                if available_models:
                    model_name = st.selectbox('Select Ollama Model', available_models)
                else:
                    st.warning('No models found. Please ensure Ollama models are loaded.')
            else:
                st.error('Ollama server is not running. Please start Ollama to use the AI agent.')
            
            temperature = st.slider(
                'Model Temperature',
                min_value=0.0,
                max_value=1.0,
                value=0.7
            )
            return model_name, temperature

    @staticmethod
    def _render_file_upload_and_query() -> Optional[str]:
        uploaded_files = st.file_uploader(
            'Upload CSV files',
            type='csv',
            accept_multiple_files=True
        )

        StreamlitUI._handle_file_upload(uploaded_files)
        StreamlitUI._handle_file_selection()

        st.header('AI Agent Query')
        query = st.text_area('Enter your query for the AI agent:', height=100)
        st.button('Submit Query', type='primary', key='submit_query')
        
        return query

    @staticmethod
    def _handle_file_upload(uploaded_files: List) -> None:
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state['uploaded_files']:
                    df = pd.read_csv(file)
                    st.session_state['uploaded_files'][file.name] = df

    @staticmethod
    def _handle_file_selection() -> None:
        if st.session_state['uploaded_files']:
            file_names = list(st.session_state['uploaded_files'].keys())
            selected_file = st.selectbox('Select File for Analysis', file_names)
            st.session_state['selected_df'] = st.session_state['uploaded_files'][selected_file]
            st.session_state['selected_df_file_name'] = selected_file

    @staticmethod
    def render_main_content() -> None:
        st.header('Data Viewer')
        if st.session_state['uploaded_files']:
            file_names = list(st.session_state['uploaded_files'].keys())
            tabs = st.tabs(file_names)
            for idx, file_name in enumerate(file_names):
                df = st.session_state['uploaded_files'][file_name]
                with tabs[idx]:
                    st.subheader(f'Data from {file_name}')
                    st.dataframe(df)

    @staticmethod
    def render_ai_response(response: AIResponse) -> None:
        st.header('AI Agent Response')
        
        if response.sql_query:
            st.subheader('Generated SQL Query')
            st.code(response.sql_query, language='sql')
            
            if response.sql_results:
                st.subheader('Query Results')
                if isinstance(response.sql_results, list):
                    st.dataframe(pd.DataFrame(response.sql_results))
                else:
                    st.write(response.sql_results)
        
        st.markdown(response.analysis)

def main():
    st.set_page_config(page_title='OllamaSQL', layout='wide')
    DataFrameManager.initialize_session_state()
    
    model_name, temperature, user_query = StreamlitUI.render_sidebar()
    StreamlitUI.render_main_content()
    
    if st.session_state.get('submit_query') and OllamaService.is_server_running():
        if st.session_state['selected_df'] is not None:
            with st.spinner('Analyzing data...'):
                response = asyncio.run(
                    OllamaService.get_ai_response(user_query, model_name, temperature)
                )
                StreamlitUI.render_ai_response(response)
        else:
            st.warning('Please select data from a CSV file for the AI agent to analyze.')
    else:
        st.warning('Please enter a query and ensure the Ollama server is running.')

if __name__ == "__main__":
    main()