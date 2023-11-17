from workflows.ingest_data import load_and_split_documents

# Non PyFlyte test to tests modules

def main():
    # slack_docs = slack_loader.load()
    # print(f"Number Slack documents: {len(slack_docs)}")
    # confluence = confluence_loader.confluence
    # pages = confluence.get_all_pages_from_space_raw("ENG")
    # print(len(pages))
    docs = load_and_split_documents()
    print(f"Number of total documents: {len(docs)}")

    # documents = get_documents_from_slack_data()
    # print(f"Number documents from library: {len(documents)}")

if __name__ == "__main__":
    main()