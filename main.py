# 라이브러리 및 모듈 가져오기
import streamlit as st  # Streamlit을 사용하여 웹 애플리케이션 생성
from pathlib import Path  # 파일 경로 작업을 위한 Path 클래스
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader  # 다양한 파일 형식에서 텍스트 추출하는 로더들
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트를 작은 청크로 나누기 위한 모듈
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 모델을 통한 텍스트 임베딩 처리
from langchain.vectorstores import FAISS  # FAISS 벡터 스토어를 통해 텍스트 검색 기능 구현
from langchain_community.callbacks import get_openai_callback  # OpenAI 응답을 받아오는 콜백
from langchain.memory import ConversationBufferMemory  # 대화 내용을 저장하는 메모리
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Streamlit에서 채팅 기록 관리
from langchain.chains import ConversationalRetrievalChain  # 문서 검색과 회신을 결합한 대화 체인
from langchain.chat_models import ChatOpenAI  # OpenAI 언어 모델 사용을 위한 모듈
from langchain.schema.messages import HumanMessage, AIMessage  # 사용자와 AI 메시지를 나타내는 스키마
import tiktoken  # 토큰화 처리를 위한 모듈
import json  # JSON 형식의 데이터 관리
import base64  # 텍스트 인코딩을 위해 Base64 사용
import speech_recognition as sr  # 음성 인식 기능을 위한 모듈
import tempfile  # 임시 파일 생성 및 관리 모듈

# 애플리케이션 실행 함수 정의
def main():
    # 페이지 설정 (Streamlit 상단 바 구성)
    st.set_page_config(page_title="에너지", page_icon="🌻")  # 웹 페이지 제목과 아이콘 설정
    st.image('energy.png')  # 상단에 이미지를 표시
    st.title("_:red[에너지 학습 도우미]_ 🏫")  # 제목 표시 (에너지 학습 도우미)
    st.header("😶주의! 이 챗봇은 참고용으로 사용하세요!", divider='rainbow')  # 주의사항 표시

    # 세션 상태 초기화
    # Streamlit 세션에서 대화 상태, 대화 기록, 처리 완료 여부 등을 초기화하여 유지
    if "conversation" not in st.session_state:  # 대화 체인 상태
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:  # 대화 기록 상태
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:  # 모델 로드 여부 상태
        st.session_state.processComplete = None
    if "voice_input" not in st.session_state:  # 음성 입력 상태
        st.session_state.voice_input = ""
    if 'messages' not in st.session_state:  # 초기 메시지 설정
        st.session_state['messages'] = [{"role": "assistant", "content": "😊"}]

    # 사이드바 구성
    with st.sidebar:
        folder_path = Path()  # 텍스트 파일이 있는 폴더 경로 (현재 경로)
        openai_api_key = st.secrets["OPENAI_API_KEY"]  # OpenAI API 키 설정 (Streamlit secrets 사용)
        model_name = 'gpt-4o-mini'  # 사용할 OpenAI 모델 이름 설정
        
        # 사이드바에 안내 메시지 및 Process 버튼
        st.text("아래의 'Process'를 누르고\n아래 채팅창이 활성화 될 때까지\n잠시 기다리세요!😊😊😊")
        process = st.button("Process", key="process_button")  # Process 버튼 추가하여 모델 초기화 및 준비
        
        # Process 버튼이 눌렸을 때 모델과 데이터 로드 및 대화 체인 설정
        if process:
            files_text = get_text_from_folder(folder_path)  # 폴더에서 텍스트 추출
            text_chunks = get_text_chunks(files_text)  # 텍스트를 분할하여 청크로 변환
            vectorstore = get_vectorstore(text_chunks)  # 텍스트 임베딩 및 벡터 스토어 생성
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, model_name)  # 대화 체인 설정
            st.session_state.processComplete = True  # 처리 완료 상태로 설정

        # 음성 입력을 받아 녹음하고 텍스트로 변환
        audio_value = st.experimental_audio_input("음성 메시지를 녹음하여 질문하세요😁.")
        
        # 음성 입력이 있을 경우, 녹음된 내용을 텍스트로 변환
        if audio_value:
            with st.spinner("음성을 인식하는 중..."):  # 처리 중임을 알리는 스피너 표시
                recognizer = sr.Recognizer()  # 음성 인식기 객체 생성
                try:
                    # 임시 오디오 파일 생성하여 음성 변환
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        temp_audio_file.write(audio_value.getvalue())
                        with sr.AudioFile(temp_audio_file.name) as source:
                            audio = recognizer.record(source)  # 오디오 데이터를 녹음
                            st.session_state.voice_input = recognizer.recognize_google(audio, language='ko-KR')  # 한국어로 변환
                    st.session_state.voice_input = st.session_state.voice_input.strip()  # 공백 제거
                except sr.UnknownValueError:
                    st.warning("음성을 인식하지 못했거나 모델을 불러오지 않았습니다. Process를 눌르고 다시 시도하세요!")
                except sr.RequestError:
                    st.warning("서버와의 연결에 문제가 있습니다. 다시 시도하세요!")
                except OSError:
                    st.error("오디오 파일을 처리하는 데 문제가 발생했습니다. 다시 시도하세요!")

        # 대화 저장 및 삭제 기능 버튼 추가
        save_button = st.button("대화 저장", key="save_button")
        if save_button:
            if st.session_state.chat_history:
                save_conversation_as_txt(st.session_state.chat_history)  # 대화 내용을 텍스트 파일로 저장
            else:
                st.warning("질문을 입력받고 응답을 확인하세요!")  # 대화 내역이 없을 경우 경고 메시지 표시
                
        clear_button = st.button("대화 내용 삭제", key="clear_button")
        if clear_button:
            # 대화 기록과 초기 메시지를 초기화
            st.session_state.chat_history = []
            st.session_state.messages = [{"role": "assistant", "content": "😊"}]
            st.experimental_set_query_params()  # 화면을 다시 로드하여 대화 내용을 초기화

    # 질문 입력 필드 (음성 입력 또는 텍스트 입력을 통한 질문)
    query = st.session_state.voice_input or st.chat_input("질문을 입력해주세요.")

    # 질문이 있을 경우, 대화 상태와 응답 처리
    if query:
        st.session_state.voice_input = ""  # 음성 입력 초기화
        try:
            st.session_state.messages.insert(0, {"role": "user", "content": query})  # 사용자 메시지 추가
            chain = st.session_state.conversation  # 대화 체인
            with st.spinner("생각 중..."):
                if chain:
                    result = chain({"question": query})  # 대화 체인을 통한 응답 생성
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']  # 대화 기록 업데이트
                    response = result['answer']  # 응답 내용 저장
                    source_documents = result.get('source_documents', [])  # 참고 문서 저장
                else:
                    response = "모델이 준비되지 않았습니다. 'Process' 버튼을 눌러 모델을 준비해주세요."  # 모델이 준비되지 않았을 경우 경고 메시지
                    source_documents = []
        except Exception as e:
            st.error("질문을 처리하는 중 오류가 발생했습니다. 다시 시도하세요.")
            response = ""
            source_documents = []

        st.session_state.messages.insert(1, {"role": "assistant", "content": response})  # AI 응답 추가

    # 대화 내역 표시
    for message_pair in (list(zip(st.session_state.messages[::2], st.session_state.messages[1::2]))):
        with st.chat_message(message_pair[0]["role"]):
            st.markdown(message_pair[0]["content"])  # 사용자 메시지 표시
        with st.chat_message(message_pair[1]["role"]):
            st.markdown(message_pair[1]["content"])  # AI 응답 메시지 표시
        if 'source_documents' in locals() and source_documents:
            with st.expander("참고 문서 확인"):
                for doc in source_documents:
                    st.markdown(doc.metadata.get('source', '출처 없음'), help=doc.page_content)  # 참고 문서 표시

# 텍스트 토큰 길이 계산 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)  # 텍스트의 토큰 길이를 반환

# 폴더에서 텍스트 추출
def get_text_from_folder(folder_path):
    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file in files:
        if file.is_file():
            if file.suffix == '.pdf':  # PDF 파일 처리
                loader = PyPDFLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.docx':  # Word 파일 처리
                loader = Docx2txtLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.pptx':  # PowerPoint 파일 처리
                loader = UnstructuredPowerPointLoader(str(file))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)  # 각 파일의 텍스트를 문서 리스트에 추가
    return doc_list

# 텍스트 분할 함수
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,  # 청크 크기
        chunk_overlap=100,  # 청크 간 중첩 길이
        length_function=tiktoken_len  # 청크 크기 계산을 위한 토큰 길이 함수
    )
    chunks = text_splitter.split_documents(text)  # 텍스트를 분할하여 청크로 변환
    return chunks

# 벡터 저장소 생성 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",  # 한국어 임베딩 모델
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # 임베딩 정규화 옵션
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)  # 청크를 벡터화하여 저장
    return vectordb

# 대화 체인 생성 함수
def get_conversation_chain(vectorstore, openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)  # OpenAI 언어 모델 설정
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='mmr'),  # 다중 문서 검색(MMR) 사용
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', input_key='question'),
        get_chat_history=lambda h: h,  # 대화 기록 가져오기
        return_source_documents=True,  # 참고 문서 반환
        verbose=True  # 자세한 로그 출력
    )
    return conversation_chain  # 대화 체인 반환

# 대화를 텍스트 파일로 저장하는 함수
def save_conversation_as_txt(chat_history):
    conversation = ""
    for message in chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"  # 역할 구분
        content = message.content
        conversation += f"유저: {role}\n내용: {content}\n\n"  # 대화 내용 작성
    
    # 텍스트 파일로 저장할 수 있는 링크 생성
    b64 = base64.b64encode(conversation.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="대화.txt">대화 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)  # 대화 다운로드 링크 생성

# 애플리케이션 실행
if __name__ == '__main__':
    main()
