import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import tiktoken
import json
import base64
import speech_recognition as sr
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel  # CLIP 모델 사용
from google.cloud import vision  # Google Cloud Vision API

# Google Cloud Vision API 클라이언트 초기화
client = vision.ImageAnnotatorClient()

# CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Google Cloud Vision API를 통한 OCR 함수
def google_vision_ocr(image):
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='PNG')
    content = image_byte_array.getvalue()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    return ""

# PDF에서 이미지 추출 및 CLIP 임베딩 생성
def extract_images_from_pdf(pdf_path):
    images_text = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # CLIP 임베딩 생성
                inputs = clip_processor(images=image, return_tensors="pt")
                image_embeddings = clip_model.get_image_features(**inputs).detach().numpy()

                # Google Vision API로 OCR
                image_text = google_vision_ocr(image)
                if image_text.strip():  # 텍스트가 있으면 추가
                    images_text.append({
                        "page": page_num + 1,
                        "image_index": img_index,
                        "text": image_text.strip(),
                        "image_embedding": image_embeddings
                    })
    return images_text

# 폴더에서 텍스트와 이미지 텍스트 추출
def get_text_from_folder(folder_path):
    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file in files:
        if file.is_file():
            if file.suffix == '.pdf':
                loader = PyPDFLoader(str(file))
                documents = loader.load_and_split()
                # 이미지 텍스트 추가
                images_text = extract_images_from_pdf(str(file))
                for image_data in images_text:
                    documents.append({
                        "content": image_data["text"],
                        "embedding": image_data["image_embedding"],
                        "metadata": {"source": file.name, "page": image_data["page"], "image_index": image_data["image_index"]}
                    })
            elif file.suffix == '.docx':
                loader = Docx2txtLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)
    return doc_list

# 텍스트 분할 함수
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

# 벡터 저장소 생성 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # FAISS에 텍스트와 이미지 벡터를 모두 저장
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# 대화 체인 생성 함수
def get_conversation_chain(vectorstore, openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='mmr'),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', input_key='question'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

# 텍스트 토큰 길이 계산 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# 대화를 텍스트 파일로 저장하는 함수
def save_conversation_as_txt(chat_history):
    conversation = ""
    for message in chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
        conversation += f"유저: {role}\n내용: {content}\n\n"
    
    b64 = base64.b64encode(conversation.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="대화.txt">대화 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)

# 애플리케이션 실행 함수 정의
def main():
    st.set_page_config(page_title="에너지 학습 도우미", page_icon="🌻")
    st.image('energy.png')
    st.title("_:red[에너지 학습 도우미]_ 🏫")
    st.header("😶주의! 이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "😊"}]

    with st.sidebar:
        folder_path = Path()
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        model_name = 'gpt-4o-mini'
        
        st.text("아래의 'Process'를 누르고\n아래 채팅창이 활성화 될 때까지\n잠시 기다리세요!😊😊😊")
        process = st.button("Process", key="process_button")
        
        if process:
            files_text = get_text_from_folder(folder_path)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, model_name)
            st.session_state.processComplete = True

        audio_value = st.experimental_audio_input("음성 메시지를 녹음하여 질문하세요😁.")
        
        if audio_value:
            with st.spinner("음성을 인식하는 중..."):
                recognizer = sr.Recognizer()
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        temp_audio_file.write(audio_value.getvalue())
                        with sr.AudioFile(temp_audio_file.name) as source:
                            audio = recognizer.record(source)
                            st.session_state.voice_input = recognizer.recognize_google(audio, language='ko-KR')
                    st.session_state.voice_input = st.session_state.voice_input.strip()
                except sr.UnknownValueError:
                    st.warning("음성을 인식하지 못했거나 모델을 불러오지 않았습니다. Process를 눌르고 다시 시도하세요!")
                except sr.RequestError:
                    st.warning("서버와의 연결에 문제가 있습니다. 다시 시도하세요!")
                except OSError:
                    st.error("오디오 파일을 처리하는 데 문제가 발생했습니다. 다시 시도하세요!")

        save_button = st.button("대화 저장", key="save_button")
        if save_button:
            if st.session_state.chat_history:
                save_conversation_as_txt(st.session_state.chat_history)
            else:
                st.warning("질문을 입력받고 응답을 확인하세요!")

        clear_button = st.button("대화 내용 삭제", key="clear_button")
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.messages = [{"role": "assistant", "content": "😊"}]
            st.experimental_set_query_params()

    query = st.session_state.voice_input or st.chat_input("질문을 입력해주세요.")

    if query:
        st.session_state.voice_input = ""
        try:
            st.session_state.messages.insert(0, {"role": "user", "content": query})
            chain = st.session_state.conversation
            with st.spinner("생각 중..."):
                if chain:
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result.get('source_documents', [])
                else:
                    response = "모델이 준비되지 않았습니다. 'Process' 버튼을 눌러 모델을 준비해주세요."
                    source_documents = []
        except Exception as e:
            st.error("질문을 처리하는 중 오류가 발생했습니다. 다시 시도하세요.")
            response = ""
            source_documents = []

        st.session_state.messages.insert(1, {"role": "assistant", "content": response})

    for message_pair in (list(zip(st.session_state.messages[::2], st.session_state.messages[1::2]))):
        with st.chat_message(message_pair[0]["role"]):
            st.markdown(message_pair[0]["content"])
        with st.chat_message(message_pair[1]["role"]):
            st.markdown(message_pair[1]["content"])
        if 'source_documents' in locals() and source_documents:
            with st.expander("참고 문서 확인"):
                for doc in source_documents:
                    st.markdown(doc.metadata.get('source', '출처 없음'), help=doc.page_content)

if __name__ == '__main__':
    main()
