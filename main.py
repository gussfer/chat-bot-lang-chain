'''
Nesta versão revisada do código:
• As funcionalidades relacionadas ao processamento de documentos foram encapsuladas na classe DocumentProcessor.
• O código de inicialização e gerenciamento da conversa foi movido para a classe ConversationManager, tornando o código mais organizado e modular.
• As variáveis globais foram eliminadas, e as dependências entre as diferentes partes do código foram tratadas por meio da passagem de objetos 
  relevantes como parâmetros ou atributos de classe.
• A função main() foi adicionada para iniciar a execução do programa. Isso torna o código mais claro e seguindo uma convenção comum em Python.
'''
import os
import PyPDF2
import json
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

class DocumentProcessor:
    """
    Classe responsável por processar os documentos PDF.
    """
    def __init__(self, pdf_directory, storage_file):
        self.pdf_directory = pdf_directory
        self.storage_file = storage_file
        self.document_contents = {}

    def extract_text_from_pdf(self, pdf_file):
        """
        Extrai texto de um arquivo PDF.
        """
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

    def update_document_contents(self):
        """
        Verifica se há novos arquivos PDF e atualiza o armazenamento.
        """
        updated = False
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.pdf_directory, filename)
                if filepath not in self.document_contents:
                    text = self.extract_text_from_pdf(filepath)
                    self.document_contents[filepath] = text
                    updated = True
        if updated:
            with open(self.storage_file, 'w') as f:
                json.dump(self.document_contents, f)

    def load_document_contents(self):
        """
        Carrega os dados do arquivo de armazenamento, se existir.
        """
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.document_contents = json.load(f)

    def get_texts(self):
        """
        Retorna uma lista de textos dos documentos.
        """
        return list(self.document_contents.values())

class ConversationManager:
    """
    Classe responsável por gerenciar a conversa com o usuário.
    """
    def __init__(self, document_processor, api_key):
        self.document_processor = document_processor
        self.api_key = api_key
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24, length_function=self.count_tokens)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key, model="text-embedding-ada-002")
        self.db = None
        self.chain = None

    def count_tokens(self, text):
        """
        Conta o número de tokens no texto.
        """
        return len(self.tokenizer.encode(text))

    def initialize(self):
        """
        Inicializa os componentes necessários para a conversa.
        """
        self.document_processor.load_document_contents()
        texts = self.document_processor.get_texts()
        chunks = self.text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(chunks, self.embeddings)
        self.chain = load_qa_chain(OpenAI(openai_api_key=self.api_key, temperature=0), chain_type="stuff")

    def start_conversation(self):
        """
        Inicia a conversa com o usuário.
        """
        context_doc = None
        while True:
            query = input("Faça uma pergunta ao normativo (digite 'sair' para encerrar): ")
            if query.lower() == "sair":
                print("Até logo!")
                break

            if "Política de Alçadas" in query:
                context_doc = "Política de Alçadas"
            elif "Outro Documento" in query:
                context_doc = "Outro Documento"

            if context_doc:
                query += f" {context_doc}"

            docs = self.db.similarity_search(query)
            input_data = {
                'input_documents': docs,
                'question': query,
            }
            output = self.chain.invoke(input=input_data)
            output_text = output['output_text']
            print("Resposta:")
            print(output_text)

def main():
    """
    Função principal para iniciar o programa.
    """
    pdf_directory = "Políticas"
    storage_file = "document_contents.json"
    api_key = os.getenv("API_KEY")

    document_processor = DocumentProcessor(pdf_directory, storage_file)
    conversation_manager = ConversationManager(document_processor, api_key)
    conversation_manager.initialize()
    conversation_manager.start_conversation()

if __name__ == "__main__":
    main()
