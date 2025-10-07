import streamlit as st
import subprocess
import psutil
import platform
import shutil
import os

st.set_page_config(page_title="⚙️ Admin Ollama", layout="centered")
st.title("🛠️ Administration des Modèles Ollama")

# 📊 Vérifier la RAM disponible
st.subheader("📊 Informations système")
ram = psutil.virtual_memory()
st.write(f"🧠 RAM disponible : {round(ram.available / (1024**3), 2)} Go")
st.write(f"💽 RAM totale : {round(ram.total / (1024**3), 2)} Go")

# ⚠️ Vérifier la VRAM GPU (si installée)
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.write(f"🎮 GPU détecté : {gpu}")
        st.write(f"🖥️ VRAM disponible : {round(vram_total, 2)} Go")
    else:
        st.info("❗ Aucun GPU CUDA détecté.")
except:
    st.info("❗ PyTorch non installé ou pas de GPU détecté.")

# 🔌 Connexion à Ollama
try:
    import ollama
    models = ollama.list()['models']
    model_names = [m['name'] for m in models]
    st.success("✅ Ollama détecté")
    st.write("📦 Modèles installés :", model_names if model_names else "Aucun modèle")
except Exception as e:
    st.error("❌ Ollama non détecté")
    st.code(str(e))
    st.stop()

# 🎯 Choisir le modèle par défaut
st.subheader("🎯 Choix du modèle par défaut")
selected_model = st.selectbox("Modèle à utiliser dans l’assistant RAG :", model_names)

if st.button("💾 Sauvegarder ce modèle par défaut"):
    with open("config_model.txt", "w") as f:
        f.write(selected_model)
    st.success(f"✅ Modèle par défaut enregistré : {selected_model}")

# 📥 Télécharger un nouveau modèle
st.subheader("⬇️ Télécharger un nouveau modèle")
new_model = st.text_input("Nom du modèle à télécharger (ex: tinyllama, mistral, llama2)")

if st.button("📦 Télécharger"):
    if new_model:
        with st.spinner(f"Téléchargement de {new_model}..."):
            result = subprocess.run(["ollama", "pull", new_model], capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"✅ Modèle '{new_model}' téléchargé avec succès.")
                st.code(result.stdout)
            else:
                st.error("❌ Échec du téléchargement :")
                st.code(result.stderr)
    else:
        st.warning("⛔ Veuillez saisir un nom de modèle.")

# 🗑️ Supprimer un modèle
st.subheader("🗑️ Supprimer un modèle installé")
model_to_delete = st.selectbox("Sélectionner un modèle à supprimer", model_names)

if st.button("❌ Supprimer le modèle"):
    with st.spinner(f"Suppression de {model_to_delete}..."):
        result = subprocess.run(["ollama", "rm", model_to_delete], capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"🗑️ Modèle '{model_to_delete}' supprimé avec succès.")
            st.code(result.stdout)
        else:
            st.error("❌ Échec de la suppression :")
            st.code(result.stderr)