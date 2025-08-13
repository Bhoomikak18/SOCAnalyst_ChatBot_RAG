import pdfplumber
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from PIL import Image
import pytesseract
import os
import io
import sys
import json
import spacy
import re
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MultiQueryRetriever
import gradio as gr

# Configure logging
logging.basicConfig(
    filename='./logs/socmate_full.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Extracting all PDF files added

# Supress Warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Begin Extracting text from the pdfs added
multiple_pdfs = './pdfs'
pdf_data_files ={}

if not os.path.exists(multiple_pdfs):
    logging.warning(f"Directory '{multiple_pdfs}' not found.")
else:
  for filename in os.listdir(multiple_pdfs):
    if filename.endswith('.pdf'):
        file_path = os.path.join(multiple_pdfs,filename)
        text_reader = PdfReader(file_path)
        text_extracted = ""

        try:
          if(len(text_reader.pages)) == 0:
            logging.warning(f"Warning: {filename} has no pages.")
            continue

          with pdfplumber.open(file_path) as pdf_files:

            for i, pdf_page in enumerate(pdf_files.pages):
              # Extract text
              text = pdf_page.extract_text()
              if text and text.strip():
                text_extracted += text + "\n"

              else:
                try:
                  logging.info(f"Running OCR for page {i} of {filename}" )
                  page_picture =  pdf_page.to_image(resolution=300)
                  image_byte = io.BytesIO()
                  page_picture.original.save(image_byte,format='PNG')
                  image_byte.seek(0)
                  ocr_text = pytesseract.image_to_string(Image.open(image_byte))
                  text_extracted += f"\n[OCR Page {i}]\n" + ocr_text + "\n"
                  logging.info(f"OCR successful for page {i} of {filename}.")
                except pytesseract.TesseractError as e_tesseract:
                   logging.error(f"OCR Tesseract Error on page {i} of {filename}: {e_tesseract}. ")
                except Exception as error_ocr:
                   logging.error(f"No text extracted via OCR from page {i} of {filename} :{error_ocr}..")

          # Storing the extracted text
          pdf_data_files[filename] = text_extracted.strip()
          logging.info(f"Text extracted from {filename}")

        except PermissionError:
          logging.error(f"Permission denied for {filename}.")
        except PdfReadError:
          logging.error(f"Error: {filename} is corrupted or password-protected.")
        except OSError:
          logging.error(f"OS error with {filename}.")

# Saving the extracted text file

output_file_path = './output/extracted_pdf_text.txt'
print_place = sys.stdout
sys.stdout = open(output_file_path, 'w', encoding='utf-8')

try:
    print("\n Extracted Content from All PDFs")
    for filename, text_extracted in pdf_data_files.items():
        print(f"\n START OF '{filename}':")
        print(text_extracted)
        print(f"\n END OF '{filename}':")
finally:
    sys.stdout.close()
    sys.stdout = print_place

if os.path.getsize(output_file_path) == 0:
    logging.error("The output file is empty. No content was written.")
    raise ValueError("The output file is empty.")
else:
  logging.info(f"\n File'{output_file_path}', contains the extracted text")

# Load the spaCy model

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

# Creating Data Processing, Semantic Chunking and Embedding Generation

output_file_path = './output/processed_pdf_text.txt'

def remove_table_of_contents(text):
    cleaned_lines = []
    for line in text.split('\n'):
        # Remove lines with many dots (like ".....")
        if re.search(r'\.{5,}', line):
            continue
        # Remove lines that look like section numbers + dots + page numbers
        if re.search(r'\d+(\.\d+)*\s+.*\.{3,}\s*\d+', line):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def load_abbreviations_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def expand_abbreviations(text, abbr_dict):
    for abbr, full_form in abbr_dict.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full_form, text, flags=re.IGNORECASE)
    return text

# Load your abbreviation dictionary from a JSON file
abbr_dict = load_abbreviations_from_json("./json/abbreviations.json")

# Semantic Chunking
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
semantic_chunker = SemanticChunker(embeddings_model,breakpoint_threshold_type='percentile')

try:

  all_documents=[] # Collecting info from all pdfs into list

  with open (output_file_path, 'w', encoding='utf-8') as file_output:
    for filename, textprocessed in pdf_data_files.items():
        try:

          # Remove OCR markers before processing
          textprocessed = re.sub(r'\[OCR Page \d+\]\n', '', textprocessed)

          # Lowercasing the data extracted
          text_processed = textprocessed.lower()

          # Remove the table of contents
          text_processed = remove_table_of_contents(text_processed)

          #Remove Table & List
          table_pattern = r'(?:\|[^|\n]*\|)+'
          matches = re.findall(table_pattern, text_processed)

          for match in matches:
              flattened = re.sub(r'[\|\s]+', ' ', match).strip()
              text_processed = text_processed.replace(match, flattened)

          #Flatten bullet points and numbered lists
          text_processed = re.sub(r'(?m)^\s*[-*+•]\s+', '', text_processed)
          text_processed = re.sub(r'(?m)^\s*\d+\.\s+', '', text_processed)

          # Removal of whitespaces
          text_processed = re.sub(r'\s+', ' ', text_processed).strip()

          # Expand abbreviations
          text_processed = expand_abbreviations(text_processed, abbr_dict)

          # Remove all characters except alphanumerics, spaces, and @ / . - :
          text_processed = re.sub(r"[^a-zA-Z0-9@/\.\-:\$&\s—]", "", text_processed)

          # This replaces any sequence of spaces, a hyphen, and more spaces with just a hyphen.
          text_processed = re.sub(r'\s*-\s*', '-', text_processed)

          # Passing  text into the SpaCy pipeline
          doc = nlp(text_processed)

          #  NER or Term Preservation
          terms = [ent.text.upper() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

         # Unique custom terms (case-insensitive match)
          custom_terms = {"DDOS", "CIA TRIAD", "PHISHING", "MITRE", "SIEM", "SOC", "CVE", "IOC", "INCIDENT RESPONSE",
           "SOC ANALYST","XDR", "APT", "INCIDENT", "RESPONSE", "SECURITY", "THREAT", "VULNERABILITY", "ATTACK", "SYSTEM", "NETWORK","ATTCK", "EDR",
          "TTP", "MFA", "SOAR", "FIREWALL", "RANSOMWARE"}

          text_lower = text_processed.lower()
          for term in custom_terms:
              if term.lower() in text_lower:
                  terms.append(term)

          terms = list(set(terms))

          # Semantic Chunking
          chunks = semantic_chunker.split_text(text_processed)
          logging.info(f"Text from '{filename}' is split into {len(chunks)} chunks")

          # Embedding Generation
          embedding_generate = embeddings_model.embed_documents(chunks)
          #print(embedding_generate[0][:5])

          # Creating the documents and adding it to a collective list

          docs = []
          for chunk in chunks:
            docs.append(Document(page_content=chunk,metadata={"source": filename, "terms": ", ".join(terms)}))
          all_documents.extend(docs)

          # Writing chunks to output file
          if chunks:
            file_output.write(f"\nSTART OF CHUNKING\n")
            #file_output.write(f"Original Text: {text_processed}\n")
            for i, chunk in enumerate(chunks):
                 file_output.write(f"CHUNK {i+1}:{chunk} \n")
            file_output.write(f"END OF CHUNKING\n")
          else:
              logging.warning(f"No chunks generated for {filename}.")

        except Exception as e:
              logging.exception(f"Error processing {filename}: {e}")

    if os.path.getsize(output_file_path) == 0:
        logging.error("The output file is empty. No content was written.")
        raise ValueError("The output file is empty.")
    logging.info(f"All cleaned texts saved to: {output_file_path}")

except Exception as e:
    logging.exception(f"Failed to save combined file: {e}")

# Creating Vector Storage Database, Integrating LLMs, Prompt, Retrival Generation

# Integrating LLM

local_model = "llama3"
o_llm = ChatOllama(model=local_model)

# Vector Database Storage uding ChromaDB with all the documents

vector_db=Chroma.from_documents(
  documents = all_documents,
  embedding = embeddings_model,
  persist_directory="./chroma_db"
)

#vector_db.persist()

# Information Retrival using MultiQueryRetriever

retriever_info = MultiQueryRetriever.from_llm(
  retriever = vector_db.as_retriever(
  search_type = "similarity", # mmr
  search_kwargs = {"k": 3}),
  llm=o_llm,
  prompt=PromptTemplate(
    input_variables=["question"],
    template="Generate 4 alternative phrasings of the following question to improve retrieval: {question}"
  )
)

# Creating custom prompt template

custom_prompt = PromptTemplate(
    input_variables = ["context","question","chat_history"],
    template = '''You are an expert assistant supporting Security Operations Center (SOC) analysts.
                Guidelines:
                1. Provide clear, accurate, and actionable responses based only on the information available in the internal documentation.
                2. Avoid vague language, hedging, or unnecessary introductions.
                3. Do not guess or make up information.
                4. If the answer cannot be found in the provided context, respond with: "Information not found."
                5. Do not start answers with phrases like "According to internal documents" or
                    or "You are an expert assistant supporting Security Operations Center (SOC) analysts" similar.
                6. Do not respond with only "yes" or "no." Always provide a brief, relevant explanation.
                7. Focus on clarity, precision, and usefulness, do not hallucinate and if an answer is avaliable search properly
                   and provide answers."
                8. If multiple questions are asked together, identify each one and provide a clear, separate answer for each.

                Conversation History:
                {chat_history}

                Context:
                {context}

                Question:
                {question}

                Answer:'''
)

# Initialize converstaion memory

conversation_memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True,
  output_key="answer"
)

# Conversational chain

qa_chain = ConversationalRetrievalChain.from_llm(
  llm = o_llm,
  retriever = retriever_info,
  memory = conversation_memory,
  return_source_documents = True,
  combine_docs_chain_kwargs = {"prompt":custom_prompt}
)

# Gradio for chatbot interaction

def chat_response(message,history):
  if not message:
    logging.warning("Empty question submitted.")
    return "Enter Question"

  try:
    # Calling conversational chain
    response = qa_chain({"question":message})
    answer = response["answer"]
    return answer
    logging.info(f"Answer: {answer}")
  except Exception as e:
      logging.exception(f"Error processing question: {e}")
      return f"Error processing question: {e}"


#Initiate the Gradio
chatbot=gr.ChatInterface(
  fn = chat_response,
  textbox=gr.Textbox(
    placeholder="Type your Question here",
    container=False,
    autoscroll=True,
    scale=7),
  title ="SOCMATE",
  theme="soft"
)

# Launch the chatbot
logging.info("Launching chatbot UI...")
chatbot.launch(share=True, inline=False)
