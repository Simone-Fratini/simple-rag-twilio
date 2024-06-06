from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_community.callbacks import get_openai_callback
import fitz  # PyMuPDF to handle PDFs
import faiss
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Carica le variabili d'ambiente da .env
load_dotenv()

# Carica il PDF
pdf_path = os.getenv('pdf_path')

# Percorso dove salvare/caricare l'indice
index_path = os.getenv('index_path')

# Twilio Credentials
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Provide your OpenAI API key directly
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI model
openai_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")

# Initialize Groq model
#groq_model = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192")

twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
# Setup Twilio client
twilio_client = Client('twilio_account_sid', 'twilio_auth_token')
 
# Global Dictionary per memorizzare la cronologia delle chat
chat_histories = {}

prompt_template = """
Sei un esperto medico che fornisce risposte basate sui foglietti illustrativi dei medicinali. Rispondi in italiano con precisione e chiarezza, utilizzando il markdown per WhatsApp dove necessario (es. *grassetto* per enfasi e _corsivo_ per termini tecnici). Le tue risposte devono:

- Devono includere il markdown whatsapp per una giusta formattazione quindi corsivo,bold,sottolieeato...
- Essere basate esclusivamente sulle informazioni del foglietto illustrativo.
- Mantenere un tono professionale.
- Evitare consigli medici diretti, indicando di consultare un medico se necessario.
- Essere concise e richiedere chiarimenti se la domanda è incompleta o ambigua.
Context: {context}
Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# LLM Chain per QA
llm_chain = LLMChain(
    llm=openai_model,
    prompt=prompt
)


# Funzione per leggere il contenuto di un PDF
def read_pdf(file_path):
    """Legge il contenuto di un PDF e lo restituisce come stringa."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Funzione per dividere il testo in blocchi di 1000 caratteri con 20 di sovrapposizione
def split_pdf_into_chunks(text):
    """Divide il testo in blocchi di 1000 caratteri con 20 di sovrapposizione."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=90)
    documents = [Document(page_content=text)]
    return text_splitter.split_documents(documents)

# Inizializza gli embeddings e il vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


#Verifica se l'indice esiste
if os.path.exists(index_path):
    # Carica l'indice da disco
    index = faiss.read_index(index_path)
    # Inizializza il docstore
    docstore = InMemoryDocstore({})
    # Inizializza vectorstore con l'indice caricato
    vectorstore = FAISS(embeddings, index, docstore, {})
    print('DATABASE VETTORIALE CARICATO!!!')
else:
    # Crea l'indice
    dimension = 1536  # Dimensione per text-embedding-ada-002
    index = faiss.IndexFlatL2(dimension)
    # Inizializza il docstore
    docstore = InMemoryDocstore({})
    # Crea vectorstore con il nuovo indice
    vectorstore = FAISS(embeddings, index, docstore, {})
    
    # Carica il PDF e aggiungi i documenti all'indice
    pdf_content = read_pdf(pdf_path)
    documents = split_pdf_into_chunks(pdf_content)
    vectorstore.add_documents(documents)
    
    # Salva l'indice su disco
    faiss.write_index(index, index_path)
    print('NESSUN DATABASE TROVATO, CREAZIONE DI UNO NUOVO!!!')


# Crea la StuffDocumentsChain
qa_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

# Crea la Conversational Retrieval Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=openai_model,
    retriever=vectorstore.as_retriever(k=3),
    return_source_documents=True
)

# Funzione per rispondere alle domande usando RAG
def answer_question(question, chat_history):
    """Risponde a una domanda utilizzando la pipeline RAG con tracciamento dei token."""
    with get_openai_callback() as cb:
        result = rag_chain({"question": question, "chat_history": chat_history})
        print(f"Tokens used for this response: {cb.total_tokens}")  # Stampa i token utilizzati
    return result['answer']

# Applicazione Flask
app = Flask(__name__)

# Endpoint Webhook
@app.route('/webhook', methods=['POST'])
def webhook():
    """Handles incoming WhatsApp messages via Twilio webhook."""
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '').strip()
    response = MessagingResponse()

    handle_message(from_number, incoming_msg, response)
    return str(response)


def RagQA(from_number, incoming_msg, response):
    if from_number not in chat_histories:
        chat_histories[from_number] = []
    history = chat_histories[from_number]

    history.append(("user", incoming_msg))
    answer = answer_question(incoming_msg, history)
    history.append(("assistant", answer))

    msg = response.message()
    msg.body(answer)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
