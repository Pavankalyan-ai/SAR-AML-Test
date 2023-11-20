import random,os,json,io,re,zipfile,tempfile
import numpy as np
import pandas as pd
import ssl
import streamlit as st
from io import BytesIO
from io import StringIO
import streamlit.components.v1 as components
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import streamlit_toggle as tog
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai 
import fitz
import docx
from gtts import gTTS
import PyPDF2
from PyPDF2 import PdfReader
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.callbacks import get_openai_callback
from usellm import Message, Options, UseLLM
from huggingface_hub import login
import pytesseract
import cv2
# from utils import text_to_docs
# import cv
# import PyPDF2
# import pytesseract
# from pdf2image import convert_from_path




@st.cache_data
def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

@st.cache_data
def pdf_to_bytes(pdf_file_):
    with open(pdf_file_,"rb") as pdf_file:
        pdf_content = pdf_file.read()
        pdf_bytes_io = io.BytesIO(pdf_content)
    return pdf_bytes_io

@st.cache_data
def read_pdf_files(path):
    pdf_files =[]
    directoty_path = path
    files = os.listdir(directoty_path)
    for file in files:
            pdf_files.append(file)
    return pdf_files


@st.cache_data
def merge_pdfs(pdf_list):
    """
    Helper function to merge PDFs
    """
    pdf_merger = PyPDF2.PdfMerger()
    for pdf in pdf_list:
        pdf_document = PyPDF2.PdfReader(pdf)
        pdf_merger.append(pdf_document)
    output_pdf = BytesIO()
    pdf_merger.write(output_pdf)
    pdf_merger.close()
    return output_pdf


@st.cache_data
def process_text(text):
    # Add your custom text processing logic here
    processed_text = text
    return processed_text



    
@st.cache_data
def merge_and_extract_text(pdf_list):
    """
    Helper function to merge PDFs and extract text
    """
    pdf_merger = PyPDF2.PdfMerger()
    for pdf in pdf_list:
        with open(pdf, 'rb') as file:
            pdf_merger.append(file)
    output_pdf = BytesIO()
    pdf_merger.write(output_pdf)
    pdf_merger.close()
    
    # Extract text from merged PDF
    merged_pdf = PyPDF2.PdfReader(output_pdf)
    all_text = []
    for page in merged_pdf.pages:
        text = page.extract_text()
        all_text.append(text)
    
    return ' '.join(all_text)

def reset_session_state():
    session_state = st.session_state
    session_state.clear()


@st.cache_data
def render_pdf_as_images(pdf_file):
    """
    Helper function to render PDF pages as images
    """
    pdf_images = []
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        img = page.get_pixmap()
        img_bytes = img.tobytes()
        pdf_images.append(img_bytes)
    pdf_document.close()
    return pdf_images

# To check if pdf is searchable
def is_searchable_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                return True

    return False



def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        all_text = []
        for page in pdf.pages:
            text = page.extract_text()
            all_text.append(text)
    return "\n".join(all_text)




# Function to add checkboxes to the DataFrame
@st.cache_data
def add_checkboxes_to_dataframe(df):
    # Create a new column 'Select' with checkboxes
    checkbox_values = [True] * (len(df) - 1) + [False]  # All True except the last row
    df['Select'] = checkbox_values
    return df

# convert scanned pdf to searchable pdf
def convert_scanned_pdf_to_searchable_pdf(input_file):
    """
     Convert a Scanned PDF to Searchable PDF

    """
    # Convert PDF to images
    print("Running OCR")
    images = convert_from_path(input_file)

    # Preprocess images using OpenCV
    for i, image in enumerate(images):
        # Convert image to grayscale
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply thresholding to remove noise
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Enhance contrast
        image = cv2.equalizeHist(image)

        # Save preprocessed image
        cv2.imwrite(f'{i}.png', image)

    # Perform OCR on preprocessed images using Tesseract
    text = ''
    for i in range(len(images)):
        image = cv2.imread(f'{i}.png')
        text += pytesseract.image_to_string(image)
    
    return text

def pytesseract_code(directory_path,fetched_files):

    tmp_dir_ = tempfile.mkdtemp()
    all_text = []
   
    #file path for uploaded files, getting files at one direc
    file_pth = []
    for uploaded_file in st.session_state.pdf_files:
        # st.write(uploaded_file)
        file_ext1 = tuple("pdf")
        file_ext2 = tuple(["png","jpeg"])
        if uploaded_file.name.endswith(file_ext1):
            file_pth_= os.path.join(tmp_dir_, uploaded_file.name)
            # st.write(file_pth_)
            with open(file_pth_, "wb") as file_opn:
                file_opn.write(uploaded_file.getbuffer())
                file_pth.append(file_pth_)
        elif uploaded_file.name.endswith(file_ext2):
            file_pth_= os.path.join(tmp_dir_, uploaded_file.name)
            file_pth.append(file_pth_)
        else:
            pass

    # For uploaded files, reading files from the created direc and using pytesseract to convert
    # This is not working for images, but only for scanned pdfs
    for file in file_pth:
        file_ext1 = tuple("pdf")
        file_ext2 = tuple(["png","jpeg"])
        if file.endswith(file_ext1):
            file_ = file.split('.',1)[0]
            if is_searchable_pdf(file)==False:
                text = convert_scanned_pdf_to_searchable_pdf(file)
                texts =  text_to_docs(text,file_)
                for i in texts:
                    all_text.append(i)
            else:
                text = extract_text_from_pdf(file)
                texts =  text_to_docs(text,file_)
                for i in texts:
                    all_text.append(i)                 
        elif file.endswith(file_ext2):
            text = convert_image_to_searchable_pdf(file)
            texts =  text_to_docs(text,file_)
            for i in texts:
                all_text.append(i)
        else:
            pass          
        
        
    #for fetched files, This is working for scanned pdf as well as images
    for fetched_pdf in fetched_files:
        file_ext1 = tuple("pdf")
        file_ext2 = tuple(["png","jpeg"])
        file = fetched_pdf.split('.',1)[0]
        if fetched_pdf.endswith(file_ext1):
            selected_file_path = os.path.join(directory_path, fetched_pdf)
            if is_searchable_pdf(selected_file_path)==False:
                text = convert_scanned_pdf_to_searchable_pdf(selected_file_path)
                texts =  text_to_docs(text,file)
                for i in texts:
                    all_text.append(i)
            else:
                file_pth = os.path.join(directory_path, fetched_pdf)
                text = extract_text_from_pdf(file_pth)
                # st.write(text)
                texts =  text_to_docs(text,file)
                for i in texts:
                    all_text.append(i)
        elif fetched_pdf.endswith(file_ext2):
            selected_file_path = os.path.join(directory_path, fetched_pdf)
            text = convert_image_to_searchable_pdf(selected_file_path)
            texts = text_to_docs(text,file)
            for i in texts:
                all_text.append(i)
        else:
            pass
    return all_text


@st.cache_data(show_spinner=False)
def convert_image_to_searchable_pdf(input_file):
    """
     Convert a Scanned PDF to Searchable PDF

    """
    # Convert PDF to images
    # images = convert_from_path(input_file)

    # # Preprocess images using OpenCV
    # for i, image in enumerate(input_file):
    # Convert image to grayscale
    image = cv2.imread(input_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply thresholding to remove noise
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Enhance contrast
    image = cv2.equalizeHist(image)

    
    file = os.path.basename(input_file)
    # Save preprocessed image
    cv2.imwrite(f'{input_file}.png', image)

    # Perform OCR on preprocessed images using Tesseract
    text = ''
    # for i in range(len(input_file)):
    image = cv2.imread(f'{input_file}.png')
    text += pytesseract.image_to_string(image)

    return text


def st_audiorec():

    # get parent directory relative to current directory
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # Create an instance of the component: STREAMLIT AUDIO RECORDER
    raw_audio_data = st_audiorec()  # raw_audio_data: stores all the data returned from the streamlit frontend
    wav_bytes = None                # wav_bytes: contains the recorded audio in .WAV format after conversion

    # the frontend returns raw audio data in the form of arraybuffer
    # (this arraybuffer is derived from web-media API WAV-blob data)

    if isinstance(raw_audio_data, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, raw_audio_data = zip(*raw_audio_data['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            raw_audio_data = np.array(raw_audio_data)  # convert to np array
            sorted_ints = raw_audio_data[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            # wav_bytes contains audio data in byte format, ready to be processed further
            wav_bytes = stream.read()

    return wav_bytes

@st.cache_data
def text_to_docs(text: str,filename) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
   
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
 
    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1
 
    # Split pages into chunks
    doc_chunks = []
 
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=50,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata = {
                "page": i + 1,"chunk": i} )
            # Add sources a metadata
            doc.metadata["source"] = filename
            doc_chunks.append(doc)
    return doc_chunks

# def convert_scanned_pdf_to_searchable_pdf(input_file, output_file):
#     """
#      Convert a Scanned PDF to Searchable PDF

#     """
#     # Convert PDF to images
#     images = convert_from_path(input_file)

#     # Preprocess images using OpenCV
#     for i, image in enumerate(images):
#         # Convert image to grayscale
#         image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

#         # Apply thresholding to remove noise
#         _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # Enhance contrast
#         image = cv2.equalizeHist(image)

#         # Save preprocessed image
#         cv2.imwrite(f'{i}.png', image)

#     # Perform OCR on preprocessed images using Tesseract
#     text = ''
#     for i in range(len(images)):
#         image = cv2.imread(f'{i}.png')
#         text += pytesseract.image_to_string(image)

#     # Add searchable layer to PDF using PyPDF2
#     pdf_writer = PyPDF2.PdfFileWriter()
#     with open(input_file, 'rb') as f:
#         pdf_reader = PyPDF2.PdfFileReader(f)
#         for i in range(pdf_reader.getNumPages()):
#             page = pdf_reader.getPage(i)
#             pdf_writer.addPage(page)
#             pdf_writer.addBookmark(f'Page {i+1}', i)

#     pdf_writer.addMetadata({
#         '/Title': os.path.splitext(os.path.basename(input_file))[0],
#         '/Author': 'Doc Manager',
#         '/Subject': 'Searchable PDF',
#         '/Keywords': 'PDF, searchable, OCR',
#         '/Creator': 'Py script',
#         '/Producer': 'EXL Service',
#     })

#     pdf_writer.addAttachment('text.txt', text.encode())

#     with open(output_file, 'wb') as f:
#         pdf_writer.write(f)

#     # Clean up temporary files
#     for i in range(len(images)):
#         os.remove(f'{i}.png')
