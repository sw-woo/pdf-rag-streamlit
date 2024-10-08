import os
import tempfile
import sys
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from streamlit_extras.buy_me_a_coffee import button
from langchain.load import dumps, loads

# 베포시 chroma db에서 sqlite3를 사용하는데 오류가 나서 추가하였다.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize buy me a coffee button
button(username="swpheus14", floating=True, width=221)

# Set the title and instructions
st.title("ChatPDF with Multiquery+hybridSearch+RagFusion")
st.write("---")
st.write("PDF 파일을 업로드하고 내용을 기반으로 질문하세요.")

# OpenAI API key input
openai_key = st.text_input('OpenAI API 키를 입력해 주세요!', type="password")

# GPT 모델 선택
model_choice = st.selectbox(
    '사용할 GPT 모델을 선택하세요:',
    ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o']
)

# File upload
uploaded_file = st.file_uploader("PDF 파일을 업로드해 주세요!", type=['pdf'])
st.write("---")

# Function to convert PDF to documents


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# Function to format documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Check if file is uploaded
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(pages)

    # Embeddings and Chroma setup
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

    # BM25 Retriever setup
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2

    # Ensemble Retriever setup
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.2, 0.8]
    )

    # Generate queries for RAG-Fusion
    template = """
    당신은 AI 언어 모델 조수입니다. 주어진 사용자 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다. 
    사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는 것이 목표입니다. 
    각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(model_name=model_choice, temperature=0, openai_api_key=openai_key)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Reciprocal Rank Fusion function
    def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results[:top_n]

    # RAG-Fusion Chain setup
    retrieval_chain_rag_fusion = generate_queries | ensemble_retriever.map(
    ) | reciprocal_rank_fusion

    # Final RAG Chain setup
    template = """다음 맥락을 바탕으로 질문에 답변하세요:

    {context}

    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name=model_choice, temperature=0,
                     openai_api_key=openai_key)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User question input
    st.header("PDF에 질문하세요!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기(ASK)'):
        with st.spinner('답변 생성 중...'):
            result = final_rag_chain.invoke(question)
            st.write(result)
