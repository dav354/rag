import streamlit as st
import requests

# Konfigurierbare Basis-URL der API
if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"

st.sidebar.header("Einstellungen")
st.session_state.api_base = st.sidebar.text_input(
    "API Basis-URL", value=st.session_state.api_base
)

API_URL = f"{st.session_state.api_base.rstrip('/')}/ask"

st.set_page_config(page_title="KG-RAG Chatbot", page_icon="🤖")
st.title("askTHWS-ChatBot")

# Chat-Historie in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

def submit_callback():
    user_input = st.session_state.user_input
    if user_input.strip():
        # Speichere User-Nachricht
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Anfrage an API
        with st.spinner("Bot denkt nach…"):
            try:
                resp = requests.post(API_URL, json={"query": user_input}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                answer_text = data.get("answer", {}).get("answer", "Keine Antwort erhalten.")
                duration = str(data.get("duration_seconds", ""))
                sources = data.get("answer", {}).get("sources", "")
            except Exception as e:
                answer_text = f"Fehler beim Abruf: {e}"
                duration = ""
                sources = ""
        # Speichere Bot-Antwort
        st.session_state.messages.append({
            "role": "bot",
            "answer": answer_text,
            "duration": duration,
            "sources": sources
        })
    # Input-Feld zurücksetzen
    st.session_state.user_input = ""

# Darstellung der bisherigen Nachrichten
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"**Du:** {msg['content']}")
    elif msg["role"] == "bot":
        # Drei Felder: Antwort, Dauer, Quellen
        st.text_area("Antwort", msg.get("answer", ""), key=f"answer_{i}", height=150)
        st.text_input("Antwortdauer (s)", msg.get("duration", ""), key=f"duration_{i}")
        st.markdown("**Quellen:**")
        for src in msg.get("sources", ""):
            if src:
                st.markdown(f"- {src}")

# Chat-Eingabe-Formular mit Callback
with st.form(key="chat_form"):
    st.text_input(
        label="Deine Nachricht",
        placeholder="Schreibe hier deine Frage...",
        key="user_input"
    )
    st.form_submit_button("Senden", on_click=submit_callback)