import os
import glob

import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm
)

from langchain.prompts import PromptTemplate

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})
folder_path = 'documentation'

def process_pdf_file(pdf_file, filename):
    pdf_reader = PdfReader(pdf_file)
    
    fileString = os.path.basename(pdf_file.name)
    logger.info(fileString)
    text = f"Pdf-Filename: {pdf_file.name}"
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(text=text)
    filename = filename or pdf_file.name

    Neo4jVector.from_texts(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="pdf_bot",
        node_label="PdfBotChunk",
        pre_delete_collection=False,
        metadatas=[{"filename": filename}]
    )

def preload_pdfs_from_folder(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as f:
            process_pdf_file(f, None)

    st.rerun()

def delete_file_and_index(file_name, index):
    query = f"MATCH (n:PdfBotChunk) WHERE n.filename = '{file_name}' DETACH DELETE n"
    index.query(query)

    #index.query("MATCH (n:PdfBotChunk) WHERE n.filename = $filename DETACH DELETE n", {"filename": file_name})
    try:
        os.remove(file_name)
        logger.info(f"File deleted: {file_name}")
        st.rerun()
    except OSError as e:
        logger.error(f"Error deleting file {file_name}: {e}")

def main():
    st.header("📄Chat with your pdf files")

    existing_index = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=url,
        username=username,
        password=password,
        index_name="pdf_bot",
        #text_node_property="info",  # Need to define if it is not default
    )
    
    with st.expander("Manage files"):
        if st.button('Reindex files'):
            existing_index.query("MATCH (n:PdfBotChunk) DETACH DELETE n")
            preload_pdfs_from_folder(folder_path)

        result = existing_index.query("MATCH (n:PdfBotChunk) RETURN n.filename")

        for x, row in enumerate(result):
            col1, col2, col3 = st.columns((1,2,1))
            col1.write(x)
            col2.write(row['n.filename'])
            button_delete = col3.empty()
            do_action = button_delete.button("Delete", key=x)
            if do_action:
                delete_file_and_index(row['n.filename'], existing_index)

        # Upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        if pdf is not None:
            with open(os.path.join(folder_path, pdf.name), 'wb') as t:
                t.write(pdf.getvalue())
            process_pdf_file(pdf, os.path.join(folder_path, pdf.name))
    
    query = st.text_input("Ask questions related to the loaded pdf files")

    if query:
        stream_handler = StreamHandler(st.empty())

        custom_prompt_template = """
        ### System:
        You are an AI assistant that follows instructions extremely well. Help as much as you can. 
        Start every answer with the Pdf-Filename or with 'No PDF found' if you cant find an answer.
        Use only the following information to answer user queries:
        Context= {context}
        Question= {question}
        """

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=custom_prompt_template)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=existing_index.as_retriever(), 
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT
            },
        )    

        qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()
