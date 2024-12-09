# main.py
from fastapi.responses import StreamingResponse
import edge_tts
import asyncio
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import shutil
from typing import List, Dict
import uvicorn
from pydantic import BaseModel
from datetime import datetime
import json

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des dossiers
UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"
METADATA_FILE = "documents_metadata.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

class DocumentMetadata:
    def __init__(self):
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_document(self, filename: str, file_type: str, chunks: int):
        doc_id = str(len(self.metadata) + 1)
        self.metadata[doc_id] = {
            "filename": filename,
            "file_type": file_type,
            "chunks": chunks,
            "upload_date": datetime.now().isoformat(),
            "status": "indexed"
        }
        self.save_metadata()
        return doc_id
    
    def remove_document(self, doc_id: str):
        if doc_id in self.metadata:
            filepath = os.path.join(UPLOAD_DIR, self.metadata[doc_id]["filename"])
            if os.path.exists(filepath):
                os.remove(filepath)
            del self.metadata[doc_id]
            self.save_metadata()
            return True
        return False
    
    def get_all_documents(self) -> Dict:
        return self.metadata

class RAGSystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key
        )
        self.model = ChatMistralAI(mistral_api_key=self.api_key)
        self.vector_store = self._load_or_create_vector_store()
        self.doc_metadata = DocumentMetadata()
        
    def _load_or_create_vector_store(self):
        if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
           return FAISS.load_local(VECTOR_STORE_DIR, self.embeddings, allow_dangerous_deserialization=True)
        return None
    
    def process_file(self, file_path: str, original_filename: str):
        """Traite un fichier PDF ou TXT et l'ajoute à la base vectorielle"""
        try:
            file_type = original_filename.split('.')[-1].lower()
            
            if file_type not in ['pdf', 'txt']:
                raise ValueError(f"Type de fichier non supporté. Seuls PDF et TXT sont acceptés.")
            
            # Sélectionne le loader approprié
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            else:  # txt
                loader = TextLoader(file_path)
            
            documents = loader.load()
            
            # Ajout de métadonnées aux documents
            for doc in documents:
                doc.metadata["source_file"] = original_filename
                doc.metadata["file_type"] = file_type
            
            # Découpage du texte
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Mise à jour ou création de la base vectorielle
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
            else:
                self.vector_store.add_documents(splits)
            
            # Sauvegarde
            self.vector_store.save_local(VECTOR_STORE_DIR)
            
            # Enregistrement des métadonnées
            doc_id = self.doc_metadata.add_document(
                original_filename,
                file_type,
                len(splits)
            )
            
            return {
                "doc_id": doc_id,
                "chunks": len(splits),
                "filename": original_filename,
                "file_type": file_type
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def query(self, question: str, context_docs: int = 4):
        """Interroge la base de connaissances"""
        if not self.vector_store:
            raise HTTPException(status_code=400, detail="Aucun document n'a été indexé")
            
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": context_docs}
        )
        
        prompt = ChatPromptTemplate.from_template("""
        Réponds à la question en te basant uniquement sur le contexte fourni.
        Si tu ne peux pas répondre avec le contexte disponible, dis-le clairement.
        Cite les sources (noms des fichiers) utilisées pour la réponse.
        
        <contexte>
        {context}
        </contexte>
        
        Question: {input}
        
        Réponse:""")
        
        document_chain = create_stuff_documents_chain(self.model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        try:
            response = retrieval_chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialisation du système RAG
from dotenv import load_dotenv
load_dotenv()

rag_system = RAGSystem(os.getenv("MISTRAL_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Endpoint pour télécharger plusieurs fichiers"""
    results = []
    
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            results.append({
                "filename": file.filename,
                "error": "Type de fichier non supporté. Seuls PDF et TXT sont acceptés."
            })
            continue
            
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Traitement du fichier
            result = rag_system.process_file(file_path, file.filename)
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "message": f"{len(results)} fichiers traités",
        "results": results
    })

@app.get("/documents")
async def get_documents():
    """Récupère la liste des documents indexés"""
    return rag_system.doc_metadata.get_all_documents()

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Supprime un document"""
    if rag_system.remove_document(doc_id):
        return {"message": f"Document {doc_id} supprimé"}
    raise HTTPException(status_code=404, detail="Document non trouvé")

@app.post("/query")
async def query(query: Query):
    """Endpoint pour poser une question"""
    try:
        response = rag_system.query(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/speak")
async def text_to_speech(text: dict):
    try:
        communicate = edge_tts.Communicate(text["text"], "fr-FR-HenriNeural")
        audio_stream = io.BytesIO()
        
        async def generate_audio():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_stream.write(chunk["data"])
            audio_stream.seek(0)
            return audio_stream
        
        audio_data = await generate_audio()
        
        return StreamingResponse(
            audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)