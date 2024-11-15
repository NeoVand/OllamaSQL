import streamlit as st
import pandas as pd
import requests
import ollama
import duckdb
import json
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from http import HTTPStatus
from time import sleep
import traceback



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
        if 'vector_indices_created' not in st.session_state:
            st.session_state['vector_indices_created'] = False
        if 'embedding_model_name' not in st.session_state:
            st.session_state['embedding_model_name'] = None
        if 'similarity_metric' not in st.session_state:
            st.session_state['similarity_metric'] = 'cosine'

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
        embedding_model_name: str,
        temperature: float,
        similarity_metric: str
    ) -> AIResponse:
        client = ollama.AsyncClient()
        
        df = st.session_state['selected_df']
        num_rows = df.shape[0]
        
        # Build detailed columns info
        columns_info_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            line = f"- '{col}': data type is {dtype}"
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                line += f", min: {min_val}, max: {max_val}, mean: {mean_val}"
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                unique_vals = df[col].dropna().unique()[:3]
                line += f", sample values: {list(unique_vals)}"
            columns_info_lines.append(line)
        
        columns_info_str = '\n'.join(columns_info_lines)
        
        # Determine the appropriate distance function
        distance_function = {
            'cosine': 'array_cosine_distance',
            'l2sq': 'array_distance'
        }[similarity_metric]

        # Placeholder for system prompt
        system_prompt = f"""
            You are a data analysis assistant. You have access to a table 'selected_df' with {num_rows} rows. The table has the following columns:

            {columns_info_str}

            Your tasks are:
            1. Generate and execute SQL queries to answer user questions about this data.
            2. Ensure that all generated SQL queries are syntactically correct and consider the SQL execution environment.
            3. If the user's question requires semantic understanding or retrieval of similar content, perform vector similarity searches using the 'embedding' column.

            **How to Perform Vector Similarity Searches in DuckDB:**

            - Use the `ORDER BY` clause with the appropriate distance function and `LIMIT` to retrieve the most similar items.
            - Example using Euclidean distance (L2-norm squared):

            SELECT * FROM selected_df
            ORDER BY array_distance(embedding, [<query_embedding>]::FLOAT[])
            LIMIT 3;

            - Example using Cosine similarity:

            SELECT * FROM selected_df
            ORDER BY array_cosine_distance(embedding, [<query_embedding>]::FLOAT[])
            LIMIT 3;

            - Replace `<query_embedding>` with the embedding vector of the user's query.

            **Process:**
            1. Embed the user's query using the embedding model to obtain `<query_embedding>`.
            2. Construct the SQL query using the appropriate distance function based on the similarity metric ('{similarity_metric}').
            3. Execute the SQL query to retrieve the most similar records.

            If an error occurs during execution, analyze the error message and adjust the query accordingly. Always return both the final SQL query and its results in your response.
            """
  # You'll need to replace this with the full prompt provided separately
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]
        
        # Initialize conversation loop
        sql_results = None
        extracted_query = None
        max_retries = 3
        for attempt in range(max_retries):
            # Get SQL query from model
            sql_response = await client.chat(
                model=model_name,
                messages=messages,
                options={"temperature": temperature},
                format='json',
                tools=[
                    {
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
                    },
                    {
                        'type': 'function',
                        'function': {
                            'name': 'embed_text',
                            'description': 'Embed text using the embedding model',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'text': {
                                        'type': 'string',
                                        'description': 'Text to embed'
                                    }
                                },
                                'required': ['text']
                            }
                        }
                    }
                ]
            )
            
            messages.append(sql_response['message'])
    
            # Execute tool calls if present
            if 'tool_calls' in sql_response['message']:
                tool_calls = sql_response['message']['tool_calls']
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    if function_name == 'execute_sql_query':
                        query_str = arguments['query']
                        extracted_query = query_str
                        query_response = SQLExecutor.execute_query(query_str)
                        sql_results = json.loads(query_response)
                        messages.append({
                            'role': 'tool',
                            'content': query_response,
                            'name': function_name
                        })
                        # Check for errors in execution
                        if isinstance(sql_results, dict) and "error" in sql_results:
                            # Inform the AI about the error
                            error_message = f"The SQL query returned an error: {sql_results['error']}"
                            messages.append({'role': 'assistant', 'content': error_message})
                            continue  # Retry with the new information
                        else:
                            break  # Successful execution
                    elif function_name == 'embed_text':
                        text = arguments['text']
                        embedding = OllamaService.get_embedding(text, embedding_model_name)
                        # Convert embedding to a string representation for SQL query
                        embedding_str = ','.join(map(str, embedding))
                        messages.append({
                            'role': 'tool',
                            'content': json.dumps({'embedding': embedding_str}),
                            'name': function_name
                        })
            else:
                break  # No tool calls, exit loop
    
            if sql_results and (not isinstance(sql_results, dict) or "error" not in sql_results):
                break  # Exit if execution is successful
    
        # Get final analysis
        final_response = await client.chat(
            model=model_name,
            messages=messages,
        )
        
        return AIResponse(
            sql_query=extracted_query,
            analysis=final_response['message']['content'],
            sql_results=sql_results
        )

    @staticmethod
    def get_embedding(text: str, model_name: str) -> List[float]:
        # Use the Ollama embed method
        embeddings = ollama.embed(model=model_name, input=[text])
        # embeddings is a list of embeddings, we need the first one
        return embeddings[0]

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
    def render_sidebar() -> tuple[Optional[str], Optional[str], float, Optional[str], str]:
        with st.sidebar:
            st.title('🦙 OllamaSQL')
            
            model_name, embedding_model_name, temperature, similarity_metric = StreamlitUI._render_model_settings()
            user_query = StreamlitUI._render_file_upload_and_query()
            
            return model_name, embedding_model_name, temperature, user_query, similarity_metric

    @staticmethod
    def _render_model_settings() -> tuple[Optional[str], Optional[str], float, str]:
        with st.expander("🛠️ Model Settings", expanded=False):
            model_name = None
            embedding_model_name = None
            similarity_metric = 'cosine'
            if OllamaService.is_server_running():
                available_models = OllamaService.get_available_models()
                if available_models:
                    model_name = st.selectbox('Select LLM Model', available_models)
                    embedding_model_name = st.selectbox('Select Embedding Model', available_models)
                    st.session_state['embedding_model_name'] = embedding_model_name  # Store in session_state
                    similarity_metric = st.selectbox('Select Similarity Metric', ['cosine', 'l2sq'])
                    st.session_state['similarity_metric'] = similarity_metric  # Store in session_state
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
            return model_name, embedding_model_name, temperature, similarity_metric

    @staticmethod
    def _render_file_upload_and_query() -> Optional[str]:
        uploaded_files = st.file_uploader(
            'Upload CSV files',
            type='csv',
            accept_multiple_files=True
        )

        StreamlitUI._handle_file_upload(uploaded_files)
        StreamlitUI._handle_file_selection()

        if st.session_state['selected_df'] is not None:
            st.subheader('Select Columns for Embedding')
            df_columns = st.session_state['selected_df'].columns.tolist()
            selected_columns = st.multiselect('Select text columns to create embeddings from:', df_columns)
            st.session_state['selected_columns'] = selected_columns

            if not st.session_state['vector_indices_created']:
                if st.button('Create Vector Indices'):
                    with st.spinner('Creating vector indices...'):
                        StreamlitUI.create_vector_indices()
                    if st.session_state['vector_indices_created']:
                        st.success('Vector indices created.')
            else:
                st.info('Vector indices already created.')

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
    def create_vector_indices() -> None:
        df = st.session_state['selected_df']
        selected_columns = st.session_state.get('selected_columns', [])
        if not selected_columns:
            st.error("No columns selected for embedding.")
            return

        total = len(df)
        progress_bar = st.progress(0)

        embeddings = []

        for idx, row in df.iterrows():
            row_texts = [str(row[col]) for col in selected_columns if pd.notnull(row[col])]
            combined_text = ' '.join(row_texts).strip()
            if not combined_text:
                combined_text = " "  # Avoid empty strings
            try:
                # Embed the text individually
                embedding_response = ollama.embed(model=st.session_state['embedding_model_name'], input=[combined_text])
                embedding = embedding_response['embeddings'][0]
                embedding = [float(value) for value in embedding]  # Convert to list of floats
                embeddings.append(embedding)
            except Exception as e:
                st.error(f"Error during embedding at index {idx}: {e}")
                return

            progress = (idx + 1) / total
            progress_bar.progress(progress)
            sleep(0.01)

        # Ensure embeddings length matches DataFrame length
        if len(embeddings) != len(df):
            st.error(f"Error: Number of embeddings ({len(embeddings)}) does not match number of DataFrame rows ({len(df)}).")
            return

        # Add embeddings to DataFrame
        df = df.copy()
        df['embedding'] = embeddings

        # Determine the embedding dimension
        embedding_length = len(embeddings[0])

        # Create a physical table in DuckDB
        conn = duckdb.connect()
        conn.execute("INSTALL 'vss';")  # Install the VSS extension
        conn.execute("LOAD 'vss';")     # Load the VSS extension

        # Define the schema for the table
        dtype_mapping = {
            'object': 'VARCHAR',
            'int64': 'BIGINT',
            'float64': 'DOUBLE',
            # Add other type mappings as needed
        }

        # Build the column definitions
        column_defs = []
        for col in df.columns:
            if col == 'embedding':
                column_defs.append(f"{col} FLOAT[{embedding_length}]")
            else:
                pandas_dtype = df[col].dtype.name
                duckdb_type = dtype_mapping.get(pandas_dtype, 'VARCHAR')  # Default to VARCHAR if type not found
                column_defs.append(f"{col} {duckdb_type}")

        # Create the table with explicit schema
        conn.execute(f"""
            CREATE TABLE selected_df (
                {', '.join(column_defs)}
            )
        """)

        # Insert data into the table
        data = df.to_records(index=False).tolist()
        conn.executemany(f"INSERT INTO selected_df VALUES ({', '.join(['?' for _ in df.columns])})", data)

        # Create vector index on the base table
        conn.execute(f"""
            CREATE INDEX hnsw_idx ON selected_df USING HNSW (embedding) WITH (metric = '{st.session_state['similarity_metric']}')
        """)

        st.session_state['selected_df'] = df  # Update the DataFrame in session state
        st.session_state['vector_indices_created'] = True
        conn.close()


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
    
    model_name, embedding_model_name, temperature, user_query, similarity_metric = StreamlitUI.render_sidebar()
    StreamlitUI.render_main_content()
    
    if st.session_state.get('submit_query') and OllamaService.is_server_running():
        if st.session_state['selected_df'] is not None:
            if not st.session_state['vector_indices_created']:
                st.warning('Please create vector indices before querying.')
            else:
                with st.spinner('Analyzing data...'):
                    response = asyncio.run(
                        OllamaService.get_ai_response(
                            user_query,
                            model_name,
                            embedding_model_name,
                            temperature,
                            similarity_metric
                        )
                    )
                    StreamlitUI.render_ai_response(response)
        else:
            st.warning('Please select data from a CSV file for the AI agent to analyze.')
    else:
        st.warning('Please enter a query and ensure the Ollama server is running.')

if __name__ == "__main__":
    main()
