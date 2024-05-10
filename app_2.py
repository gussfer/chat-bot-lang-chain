import os
import PyPDF2
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import dotenv

# Configuração da API do OpenAI
dotenv.load_dotenv(dotenv.find_dotenv())
API_KEY = os.getenv("API_KEY")

# Lendo o arquivo PDF e extraindo o texto
texto = ""
with open("Políticas\Política de Alçadas - Algar S.A-V2-01jan2023.pdf", 'rb') as arquivo:
    leitor_pdf = PyPDF2.PdfReader(arquivo)
    for pagina in leitor_pdf.pages:
        texto += pagina.extract_text()

# Tokenização do texto
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Função para contar quantidade de tokens do documento
def count_tokens(text: str):
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([texto])

# Embeddings = transformar palavras em números
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-ada-002")
db = FAISS.from_documents(chunks, embeddings)

query = input("Faça uma pergunta ao normativo: ")
docs = db.similarity_search(query)

# Verificando o modelo carregado
chain = load_qa_chain(OpenAI(openai_api_key=API_KEY, temperature=0), chain_type="stuff")

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