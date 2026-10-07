from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


load_dotenv()


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
    st.title("RAG Chatbot")
    st.info("RAG 파이프라인을 연결할 준비가 되었습니다.")


if __name__ == "__main__":
    main()
