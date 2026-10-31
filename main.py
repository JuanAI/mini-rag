import getpass
import os
import uuid

from rag import RAGBot
import telemetry

if __name__ == "__main__":
    telemetry.setup_telemetry()
    telemetry.setup_langfuse()
    # One CLI run is one conversation: Langfuse groups its traces by session.
    session_id = str(uuid.uuid4())
    user_id = os.environ.get("RAGBOT_USER_ID") or getpass.getuser()

    print("Initialising RAGBot...\n")
    rag = RAGBot()
    print("Reading and embedding data...\n")
    rag.read_and_embed_data(folder_path="data/")

    print(
        "Hi, I'm your RAGBot. Ask me anything about Jupiter, Ada Lovelace, or CRISPR\n"
    )

    try:
        while True:
            try:
                question = input("Me:")
            except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C ends the session
                print()
                break
            try:
                response = rag.ask(question=question, session_id=session_id, user_id=user_id)
            except Exception as exc:  # a failed API call shouldn't end the session
                print(f"RAGBot: Sorry, that request failed: {exc}\n")
                continue
            print(f"RAGBot: {response}\n")
    finally:
        telemetry.flush_telemetry()
