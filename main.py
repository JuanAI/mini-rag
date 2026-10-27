from rag import RAGBot

if __name__ == "__main__":
    print("Initialising RAGBot...\n")
    rag = RAGBot()
    print("Reading and embedding data...\n")
    rag.read_and_embed_data(folder_path="data/")

    print(
        "Hi, I'm your RAGBot. Ask me anything about Jupiter, Ada Lovelace, or CRISPR\n"
    )

    while True:
        try:
            question = input("Me:")
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C ends the session
            print()
            break
        try:
            response = rag.ask(question=question)
        except Exception as exc:  # a failed API call shouldn't end the session
            print(f"RAGBot: Sorry, that request failed: {exc}\n")
            continue
        print(f"RAGBot: {response}\n")
