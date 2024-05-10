import os
import PyPDF2
import json
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import dotenv

# Configuração da API do OpenAI
dotenv.load_dotenv(dotenv.find_dotenv())
API_KEY = os.getenv("API_KEY")

# Diretório contendo os arquivos PDF
pdf_directory = "Políticas"
storage_file = "document_contents.json"

# Função para ler e extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        leitor_pdf = PyPDF2.PdfReader(file)
        texto = ""
        for pagina in leitor_pdf.pages:
            texto += pagina.extract_text()
        return texto

# Se já existe um arquivo de armazenamento, carregue os dados dele
if os.path.exists(storage_file):
    with open(storage_file, 'r') as f:
        document_contents = json.load(f)
else:
    document_contents = {}

# Verificar se há novos arquivos PDF no diretório e atualizar o armazenamento se necessário
updated = False
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_directory, filename)
        if filepath not in document_contents:
            texto = extract_text_from_pdf(filepath)
            document_contents[filepath] = texto
            updated = True

# Se houve atualização, salvar os novos dados no arquivo de armazenamento
if updated:
    with open(storage_file, 'w') as f:
        json.dump(document_contents, f)

# Tokenização do texto
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Função para contar quantidade de tokens do documento
def count_tokens(text: str):
    return len(tokenizer.encode(text))

# Criar documentos a partir do texto de cada PDF
texts = list(document_contents.values())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)
chunks = text_splitter.create_documents(texts)

# Embeddings = transformar palavras em números
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-ada-002")
db = FAISS.from_documents(chunks, embeddings)

# Verificando o modelo carregado
chain = load_qa_chain(OpenAI(openai_api_key=API_KEY, temperature=0), chain_type="stuff")

# Loop de conversa
while True:
    query = input("Faça uma pergunta ao normativo (digite 'sair' para encerrar): ")
    if query.lower() == "sair":
        print("Até logo!")
        break

    docs = db.similarity_search(query)

    # Adjust your code to include an 'input' dictionary
    input_data = {
        'input_documents': docs,
        'question': query,
    }

    # Chamando o método invoke com o argumento "input" explicitamente
    output = chain.invoke(input=input_data)

    output_text = output['output_text']
    print("Resposta:")
    print(output_text)
