from dotenv import load_dotenv
from langchain_core import __version__ as core_version
from langchain_google_genai import ChatGoogleGenerativeAI

# from langgraph import __version__ as lg_version
from langchain_groq import ChatGroq

load_dotenv()

print(f"langchain-core version: {core_version}")
# print(f"langgraph version: {lg_version}")


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("What is the capital of France?")
    print(f"Response from ChatGroq: {response}")

    llm_google = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    response = llm_google.invoke("What is the capital of France?")
    print(f"Response from ChatGoogle: {response}")


if __name__ == "__main__":
    main()
