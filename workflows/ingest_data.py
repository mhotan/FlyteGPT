import os
import pickle
import re
import tempfile
from typing import Any

import boto3
from langchain.document_loaders import (ConfluenceLoader, GitLoader,
                                        ReadTheDocsLoader,
                                        SlackDirectoryLoader)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import (CharacterTextSplitter,
                                     PythonCodeTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores.faiss import FAISS

# Create a Boto3 S3 client
s3 = boto3.client('s3')

# s3://union-union-compute-us-east-2-load-test-aws/hackathon_artifacts/
# Specify the bucket and object key
bucket_name = 'union-union-compute-us-east-2-load-test-aws'
object_key = 'hackathon_artifacts/2022-11-17_to_2023-11-17_union.zip'

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

def get_documents_from_confluence_data():
    url = "https://unionai.atlassian.net/wiki"
    confluence_loader = ConfluenceLoader(
        url,
        username="mike@union.ai",
        api_key="TODO"
    )
    space_key = "ENG"
    raw_documents = confluence_loader.load(space_key=space_key)
    text_splitter = CharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(
        f"Loaded {len(documents)} documents from Confluence URL: {url} Space: {space_key}")
    return documents


def python_path_filter(path: str):
    pattern = r'^(?!.*mock)(?!.*test).*\.(py)$'
    regex = re.compile(pattern)
    match = bool(regex.match(path))
    return match

def get_documents_from_python_data(path):
    loader = GitLoader(repo_path=path, branch="master", file_filter=python_path_filter)
    raw_documents = loader.load()
    text_splitter = PythonCodeTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(f"Loaded {len(documents)} documents from {path}")
    return documents

class GolangTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Golang (*.go) syntax."""

    def __init__(self, **kwargs: Any):
        """Initialize a GolangTextSplitter."""
        separators = [
            # Split along function definitions
            "\nfunc ",
            "\nvar ",
            "\nconst ",
            "\ntype ",
            # Split along control flow statements
            "\nif ",
            "\nfor ",
            "\nswitch ",
            "\ncase ",
            # Split by the normal type of lines
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)

def golang_path_filter(path: str):
    pattern = r'^(?!.*mock)(?!.*test).*\.(go)$'
    regex = re.compile(pattern)
    match = bool(regex.match(path))
    return match

def get_documents_from_golang_data(path):
    loader = GitLoader(repo_path=path, branch="master", file_filter=golang_path_filter)
    raw_documents = loader.load()
    text_splitter = GolangTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(f"Loaded {len(documents)} documents from {path}")
    return documents

class RSTTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along reStructuredText (*.rst) syntax."""

    def __init__(self, **kwargs: Any):
        """Initialize an RSTTextSplitter."""
        separators = [
            # Split along section titles
            "\n===\n",
            "\n---\n",
            "\n***\n",
            # Split along directive markers
            "\n.. ",
            # Split by the normal type of lines
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)

def rst_path_filter(path: str):
    pattern = r'^(?!.*mock)(?!.*test).*\.(rst)$'
    regex = re.compile(pattern)
    match = bool(regex.match(path))
    return match

def get_documents_from_rst_data(path):
    loader = GitLoader(repo_path=path, branch="master", file_filter=rst_path_filter)
    raw_documents = loader.load()
    text_splitter = RSTTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(f"Loaded {len(documents)} documents from {path}")
    return documents

class ProtoTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Protocol Buffer (*.proto) syntax."""

    def __init__(self, **kwargs: Any):
        """Initialize a ProtoTextSplitter."""
        separators = [
            # Split along message definitions
            "\nmessage ",
            # Split along service definitions
            "\nservice ",
            # Split along enum definitions
            "\nenum ",
            # Split along option definitions
            "\noption ",
            # Split along import statements
            "\nimport ",
            # Split along syntax declarations
            "\nsyntax ",
            # Split by the normal type of lines
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)

def proto_path_filter(path: str):
    pattern = r'^(?!.*mock)(?!.*test).*\.(proto)$'
    regex = re.compile(pattern)
    match = bool(regex.match(path))
    return match

def get_documents_from_proto_data(path):
    loader = GitLoader(repo_path=path, branch="master", file_filter=proto_path_filter)
    raw_documents = loader.load()
    text_splitter = ProtoTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    print(f"Loaded {len(documents)} documents from {path}")
    return documents


def load_and_split_documents():
    slack_documents = get_documents_from_slack_data()
    confluence_documents = get_documents_from_confluence_data()
    return slack_documents + confluence_documents

def embed_and_vectorize_documents(documents):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

if __name__ == "__main__":
    load_and_split_documents()