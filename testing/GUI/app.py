# File: app.py
# Description: v3.0 - Fehler behoben und Anzeige auf reinen JSON-Kontext reduziert.

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
    user_input = st.session_state.get("user_input")
    if user_input and user_input.strip():
        # Speichere User-Nachricht
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Anfrage an API
        with st.spinner("Bot denkt nach…"):
            try:
                resp = requests.post(API_URL, json={"query": user_input}, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                answer_data = data.get("answer", {})
                answer_text = answer_data.get("answer", "Keine Antwort erhalten.")
                sources = answer_data.get("sources", [])
                duration = str(data.get("duration_seconds", ""))

            except requests.exceptions.RequestException as e:
                answer_text = f"Fehler bei der API-Anfrage: {e}"
                duration = ""
                sources = []
            except Exception as e:
                answer_text = f"Ein unerwarteter Fehler ist aufgetreten: {e}"
                duration = ""
                sources = []

        # Speichere Bot-Antwort
        st.session_state.messages.append({
            "role": "bot",
            "answer": answer_text,
            "duration": duration,
            "sources": sources
        })

    # ENTFERNT: Die folgende Zeile hat den Fehler verursacht und wird nicht benötigt.
    # st.session_state.user_input = ""


# --- Darstellung der Chat-Nachrichten ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Für Bot-Nachrichten, füge die Antwort und den Kontext-Expander hinzu
        if msg["role"] == "bot":
            st.markdown(msg.get("answer", ""))

            sources_list = msg.get("sources", [])
            if sources_list:
                # Expander nur für den reinen Kontext, ohne Metadaten
                with st.expander("Verwendeten Kontext anzeigen"):
                    # Iteriere durch die JSON-Objekte der Quellen
                    for src in sources_list:
                        if isinstance(src, dict):
                            # Extrahiere und zeige nur den Inhalt ("content")
                            content = src.get("content", "Kein Inhalt verfügbar.")
                            st.markdown("---")
                            st.markdown(content)
                        else:
                            # Fallback für unerwartete Formate
                            st.markdown("---")
                            st.text(str(src))

# Chat-Eingabe-Formular am unteren Rand
# Der on_submit-Callback kümmert sich um die Logik.
st.chat_input("Stelle deine Frage...", key="user_input", on_submit=submit_callback)