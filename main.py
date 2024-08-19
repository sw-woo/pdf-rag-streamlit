# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from streamlit_extras.buy_me_a_coffee import button
import os
import tempfile
import streamlit as st
# from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# 베포시 chroma db에서 sqlite3를 사용하는데 오류가 나서 추가하였다.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from langchain.embeddings import OpenAIEmbeddings


button(username="swpheus14", floating=True, width=221)

# 제목
st.title("ChatPDF")
st.write("---")

# OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    print(texts[:5])

    # Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # Load into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Define the prompt template
    prompt_template = """
    질문에 대답하는 조수 역할을 합니다. 다음에 제공된 맥락을 사용하여 질문에 답하세요. 답을 모르는 경우 그냥 모른다고 말하세요. 최대 세 문장을 사용하여 답변을 간결하게 작성하세요.

    질문: {question}

    맥락: {context}

    답변을 작성할 때 다음 지침을 따르세요:
    1. 주어진 맥락 정보에 있는 내용만을 사용하여 답변하세요.
    2. 맥락 정보에 없는 내용은 답변에 포함하지 마세요.
    3. 질문과 관련이 없는 정보는 제외하세요.
    4. 답변은 간결하고 명확하게 작성하세요.


    답변:
    """

    prompt = PromptTemplate(
        input_variables=["question", "context"], template=prompt_template)

    # Question input
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-4o-mini",
                             temperature=0, openai_api_key=openai_key)

            qa_chain = (
                {
                    "context": db.as_retriever(search_kwargs={"k": 3}) | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            result = qa_chain.invoke(question)
            print(result)
            st.write(result)
