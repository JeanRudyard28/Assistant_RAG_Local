import streamlit as st
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Configuration
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

st.set_page_config(
    page_title="Assistant RAG Local",
    page_icon="🧠",
    layout="wide"
)

# Chemins constants
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
MODELS_DIR = BASE_DIR / "models"

def get_model_choice():
    """Récupère le modèle choisi depuis le fichier de configuration"""
    config_file = BASE_DIR / "config_model.txt"
    try:
        with open(config_file, "r") as f:
            return f.read().strip()
    except:
        return "tinyllama"

def check_system_health():
    """Vérifie l'état du système"""
    health_status = {}
    
    # Vérifier Ollama
    try:
        import ollama
        models = ollama.list()
        health_status['ollama'] = {
            'status': '✅ Connecté',
            'models': [m['name'] for m in models['models']] if models['models'] else []
        }
    except Exception as e:
        health_status['ollama'] = {'status': f'❌ Erreur: {str(e)}', 'models': []}
    
    # Vérifier la base vectorielle
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        health_status['vectorstore'] = '✅ Trouvée'
    else:
        health_status['vectorstore'] = '❌ Manquante'
    
    # Vérifier les documents
    if DOCUMENTS_DIR.exists() and any(DOCUMENTS_DIR.glob("*.pdf")):
        pdf_count = len(list(DOCUMENTS_DIR.glob("*.pdf")))
        health_status['documents'] = f'✅ {pdf_count} PDF(s) trouvé(s)'
    else:
        health_status['documents'] = '❌ Aucun PDF trouvé'
    
    return health_status

def main():
    st.title("🧠 Assistant RAG Local & Offline")
    st.markdown("Posez des questions sur vos documents PDF - **100% local et privé**")
    
    # Vérification du système
    health_status = check_system_health()
    
    with st.sidebar:
        st.header("⚙️ État du système")
        
        for component, status in health_status.items():
            if component == 'ollama':
                st.write(f"**Ollama:** {status['status']}")
                if status['models']:
                    st.write(f"**Modèles:** {', '.join(status['models'])}")
            else:
                st.write(f"**{component.title()}:** {status}")
        
        # Lien vers l'administration
        st.markdown("---")
        st.page_link("pages/1_Admin_Ollama.py", label="🛠️ Administration Ollama", icon="⚙️")
        
        # Statistiques
        st.markdown("---")
        st.header("📊 Statistiques")
        if VECTORSTORE_DIR.exists():
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore = FAISS.load_local(
                    str(VECTORSTORE_DIR), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                st.write(f"**Documents indexés:** {vectorstore.index.ntotal}")
            except:
                st.write("**Documents indexés:** Chargement...")
    
    # Interface principale
    try:
        if health_status['vectorstore'] == '✅ Trouvée' and health_status['ollama']['status'] == '✅ Connecté':
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.llms import Ollama
            
            with st.spinner("🔄 Chargement du système RAG..."):
                # Modèle d'embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Chargement de la base FAISS
                vectorstore = FAISS.load_local(
                    str(VECTORSTORE_DIR), 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # Connexion à Ollama
                model_choice = get_model_choice()
                llm = Ollama(model=model_choice)
            
            st.success(f"✅ Système RAG chargé avec le modèle: {model_choice}")
            
            # Interface de question
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input(
                    "❓ Posez votre question :", 
                    placeholder="Ex: Quelles sont les tendances technologiques en 2024 ?"
                )
            with col2:
                k_slider = st.slider("Nombre de sources", 1, 5, 3)
            
            if query:
                with st.spinner("🔍 Recherche dans les documents..."):
                    # Recherche
                    docs = vectorstore.similarity_search(query, k=k_slider)
                    
                    # Construction du prompt
                    context = "\n\n".join([doc.page_content for doc in docs])
                    prompt = f"""Réponds à cette question en français en utilisant le contexte suivant:

CONTEXTE:
{context}

QUESTION: {query}

RÉPONSE:"""
                    
                    # Génération
                    response = llm.invoke(prompt)
                    
                    # Affichage
                    st.subheader("📢 Réponse :")
                    st.write(response)
                    
                    st.subheader("📚 Sources utilisées :")
                    for i, doc in enumerate(docs):
                        with st.expander(f"Source {i+1}: {os.path.basename(doc.metadata.get('source', 'Inconnu'))}"):
                            st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                            st.write(f"**Contenu:** {doc.page_content}")
        else:
            st.error("❌ Le système n'est pas prêt. Vérifiez l'état des composants dans la sidebar.")
            
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        st.info("💡 Essayez de recréer la base vectorielle avec le script de réparation.")

if __name__ == "__main__":
    main()