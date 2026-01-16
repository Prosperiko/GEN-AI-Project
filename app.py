import streamlit as st
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
#from langchain.chains import LLMChain  # <--- FIXED IMPORT
from langchain_openai import ChatOpenAI

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuMind AI", page_icon="📄")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Get your free API Token from OpenAI platform settings.")
hf_api_key = st.sidebar.text_input("OpenAI API Token", type="password")

# --- FUNCTIONS ---

def extract_text_from_pdf(file):
    """
    Extracts text from PDF.
    1. Tries standard text extraction.
    2. If text is empty (scanned image), uses OCR (Tesseract).
    """
    text = ""
    try:
        # Method 1: Standard Text Extraction
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Method 2: OCR (if standard extraction failed or was too short)
        if len(text) < 50:
            st.info("⚠️ Scanned document detected. Engaging OCR (this may take a moment)...")
            # Reset file pointer to read bytes for image conversion
            file.seek(0)
            images = convert_from_bytes(file.read())
            
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img)
            
            text = ocr_text
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
        
    return text

def get_llm_response(text_context, user_question, api_key):
    repo_id = "google/flan-t5-large"

    template = """
    You are a helpful and safe academic assistant.
    Use the following context to answer the question.

    RULES:
    1. Answer ONLY from the context.
    2. If not found, say: "I cannot find that information in the document."
    3. Keep answers concise.
    4. They have a max token of 512, so whatever you answer must fit within that limit.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    try:
        # llm = HuggingFaceEndpoint(
        #     repo_id=repo_id,
        #     huggingfacehub_api_token=api_key,
        #     temperature=0.3,
        #     max_new_tokens=512,
        #     timeout=300
        # )
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key= api_key,
            temperature=0.3,
            max_tokens=512
        )

        # ✅ NEW LangChain pattern (Runnable)
        chain = prompt | llm

        response = chain.invoke({
            "context": text_context,
            "question": user_question
        })

        return response.content

    except Exception as e:
        return f"Error connecting to model: {repr(e)}"

# --- MAIN APP UI ---

st.title("📄 DocuMind: AI PDF Scanner")
st.markdown("Upload a PDF (Text or Image/Scanned) to summarize it or ask questions.")

if not hf_api_key:
    st.warning("Please enter your OpenAI API Token in the sidebar to proceed.")
else:
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Scanning and extracting text..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            
        if extracted_text and len(extracted_text) > 10:
            st.success("Document scanned successfully!")
            
            with st.expander("View Scanned Text"):
                st.text_area("Extracted Content", extracted_text, height=200)

            # --- QUESTION ANSWERING ---
            st.divider()
            st.subheader("🤖 Ask the AI")
            
            # Suggestion chips
            col1, col2 = st.columns(2)
            if col1.button("Summarize this document"):
                with st.spinner("Analyzing..."):
                    answer = get_llm_response(extracted_text, "Summarize the main points of this document.", hf_api_key)
                    st.write(answer)
            
            user_question = st.text_input("Or type your own question:")
            
            if user_question:
                with st.spinner("Thinking..."):
                    answer = get_llm_response(extracted_text, user_question, hf_api_key)
                    st.write("### Answer:")
                    st.write(answer)
        else:
            st.error("Could not extract text. The file might be empty or too blurry.")