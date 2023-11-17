import tempfile
import time
from typing import List

import boto3
import torch
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (ConfluenceLoader, GitLoader,
                                        ReadTheDocsLoader,
                                        SlackDirectoryLoader)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import (CharacterTextSplitter,
                                     PythonCodeTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores.faiss import FAISS

_template = """Given the following chat history and a follow up question, rephrase the follow up question to be a standalone question.
You can assume that the question is about working within Union.

Chat History:
{chat_history}
Follow Up Question:
{question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an Employee at Union.
You are given the following extracted parts of the context and a question. Provide a conversational answer in a concise and clear manner. Attach a link if neccessary.
Please answer based on the question.

Question: {question}
=========
Context:
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])

# Create a Boto3 S3 client
s3 = boto3.client('s3')

# s3://union-union-compute-us-east-2-load-test-aws/hackathon_artifacts/
# Specify the bucket and object key
bucket_name = 'union-union-compute-us-east-2-load-test-aws'
# object_key = 'hackathon_artifacts/2022-11-17_to_2023-11-17_union.zip'
object_key = 'hackathon_artifacts/2023-10-17_to_2023-11-15_union.zip'

def get_chain(openai_api_key: str, vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(search_kwargs={"k": 20}),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT},
        # verbose=True,
    )
    return qa_chain

def get_documents_from_slack_data(object_key=object_key):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    # Download the file from S3 to the temporary file
    s3.download_file(bucket_name, object_key, temp_file.name)

    loader = SlackDirectoryLoader(zip_path=temp_file.name)
    raw_documents = loader.load()
    text_splitter = CharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(
        f"Loaded {len(documents)} documents from s3://{bucket_name}/{object_key}")
    return documents

def get_documents_from_confluence_data(confluence_api_key: str):
    url = "https://unionai.atlassian.net/wiki"
    confluence_loader = ConfluenceLoader(
        url,
        username="mike@union.ai",
        api_key=confluence_api_key
    )
    space_key = "ENG"
    raw_documents = confluence_loader.load(space_key=space_key)
    text_splitter = CharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(
        f"Loaded {len(documents)} documents from Confluence URL: {url} Space: {space_key}")
    return documents


def load_and_split_documents(confluence_api_key: str):
    slack_documents = get_documents_from_slack_data()
    confluence_documents = get_documents_from_confluence_data(
        confluence_api_key=confluence_api_key)
    return slack_documents + confluence_documents


def embed_and_vectorize_documents(documents):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

@task(
    requests=Resources(cpu="4", mem="32Gi", gpu="1"),
    limits=Resources(cpu="4", mem="32Gi", gpu="1"),
    cache=True,
    cache_version="0.0.0"
)
def ingest(confluence_api_key: str) -> FlyteFile:
    documents = load_and_split_documents(confluence_api_key=confluence_api_key)
    vectorstore = embed_and_vectorize_documents(documents)
    torch.save(vectorstore, "./vectorstore.pt")

    return FlyteFile("./vectorstore.pt")

@task(
    requests=Resources(cpu="4", mem="32Gi", gpu="1"),
    limits=Resources(cpu="4", mem="32Gi", gpu="1"),
)
def query(openai_api_key: str, vectorstore_file: FlyteFile, questions: List[str]):
    vectorstore = torch.load(vectorstore_file.download())
    qa_chain = get_chain(openai_api_key, vectorstore)
    chat_history = []
    for question in questions:
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("### Human:")
        print(question)
        print("### AI:")
        print(result["answer"] + "\n")

@workflow
def workflow(confluence_api_key: str,
             openai_api_key: str):
    vectorstore_file = ingest(confluence_api_key=confluence_api_key)
    query(openai_api_key=openai_api_key,
          vectorstore_file=vectorstore_file,
          questions=[
              "What are On-Call responsibilities?",
              "How do I escalate an incident?",
              "How many alerts fired in October?"])
