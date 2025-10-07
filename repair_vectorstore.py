import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def repair_vectorstore():
    print("🔧 Réparation de la base vectorielle...")
    
    BASE_DIR = Path(__file__).parent
    DOCUMENTS_DIR = BASE_DIR / "documents"
    VECTORSTORE_DIR = BASE_DIR / "vectorstore"
    MODELS_DIR = BASE_DIR / "models"
    
    # Supprimer l'ancien index corrompu
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
        print("🗑️ Ancien index vectorstore supprimé")
    
    # Vérifier les documents
    if not DOCUMENTS_DIR.exists():
        print("❌ Dossier 'documents' introuvable")
        DOCUMENTS_DIR.mkdir()
        print("✅ Dossier 'documents' créé")
        return
    
    # Initialiser les embeddings
    print("📥 Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Charger tous les documents
    print("📄 Chargement des documents...")
    documents = []
    
    for file in DOCUMENTS_DIR.iterdir():
        try:
            if file.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = file.name
                documents.extend(docs)
                print(f"✅ PDF chargé: {file.name} ({len(docs)} pages)")
                
            elif file.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = file.name
                documents.extend(docs)
                print(f"✅ DOCX chargé: {file.name} ({len(docs)} sections)")
                
        except Exception as e:
            print(f"❌ Erreur avec {file.name}: {e}")
    
    if documents:
        # Découpage des documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Créer la nouvelle base vectorielle
        print(f"🔨 Création de la base vectorielle avec {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(VECTORSTORE_DIR))
        print("✅ Base vectorielle recréée avec succès!")
        print(f"📊 Documents indexés: {len(chunks)} chunks")
    else:
        print("⚠️ Aucun document trouvé dans le dossier 'documents'")
        print("💡 Ajoutez vos fichiers PDF/DOCX dans le dossier 'documents/'")

if __name__ == "__main__":
    repair_vectorstore()