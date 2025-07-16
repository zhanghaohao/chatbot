import streamlit as st
import requests
import json

def post_data(api_endpoint, data):
    try:
        response = requests.post(api_endpoint, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

def chat_page():
    st.title("Chatbot")
    api_endpoint = "http://localhost:9092/chat"

    query = st.text_input("Query")
    namespace = st.text_input("Namespace (Optional)", value=None)
    model_selector = st.selectbox("Chat Model", 
    ["gpt-4-0613", "gpt-4o-2024-05-13"]
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    source_documents = st.slider("Number of Source Documents", 1, 10, 5)

    
    if api_endpoint:
        if st.button("Send"):
            data = {
                "query": str(query),
                "chat_history": [],
                "model": model_selector,
                "temperature": temperature,
                "vector_fetch_k": source_documents,
                "namespace": namespace if namespace else None
            }

            data = post_data(api_endpoint, data)

            if data:
                st.markdown(data['response']['answer'])
                st.header("Source Documents")
                doc_num_init = 1
                for msg in data['response']['source_documents']:
                    st.markdown(f"""### Document **[{doc_num_init}]**: \n__{msg.get('metadata').get('source')}__\n\n*Page Content:*\n\n```{msg.get('page_content')}```""")
                    doc_num_init += 1
            else:
                st.warning("Failed to fetch data from the API endpoint. Please check the endpoint URL.")

def ingest_page():
    st.title("Document Ingestion")
    FASTAPI_URL = "http://localhost:9092/ingest"  

    uploaded_files = st.file_uploader("Upload Documents(support format of pdf, txt and docs)", accept_multiple_files=True)
    namespace = st.text_input("Namespace (Optional)")

    if st.button("Ingest Documents"):
        if uploaded_files:
            try:
                files = [("files", file) for file in uploaded_files]
                payload = {"namespace": namespace}
                response = requests.post(FASTAPI_URL, files=files, data=payload)
                
                if response.status_code == 200:
                    st.success("Documents ingested successfully!")
                else:
                    st.error(f"Failed to ingest documents. Error: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload at least one document.")

def main():
    st.sidebar.title("Navigation")
    tabs = ["Ingestion", "Chat"]
    selected_tab = st.sidebar.radio("Go to", tabs)

    if selected_tab == "Chat":
        chat_page()
    elif selected_tab == "Ingestion":
        ingest_page()

if __name__ == "__main__":
    main()
