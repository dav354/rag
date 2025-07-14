# File: app.py
# Description: v4.0 - Angepasst an die flache API-Struktur (answer: string, sources: json) und Fehler in der Chat-Anzeige behoben.

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

                # --- KORRIGIERTE DATEN-EXTRAKTION ---
                # Die API liefert jetzt eine flache Struktur, wie vom retrieval.py vorgegeben.
                answer_text = data.get("answer", "Keine Antwort erhalten.")
                sources = data.get("sources", [])  # "sources" ist der neue Name für den Kontext
                duration = str(data.get("duration_seconds", ""))

            except requests.exceptions.RequestException as e:
                answer_text = f"Fehler bei der API-Anfrage: {e}"
                duration = ""
                sources = []
            except Exception as e:
                answer_text = f"Ein unerwarteter Fehler ist aufgetreten: {e}"
                duration = ""
                sources = []

        # Speichere Bot-Antwort. Das Dictionary enthält "answer", aber kein "content".
        st.session_state.messages.append({
            "role": "bot",
            "answer": answer_text,
            "duration": duration,
            "sources": sources
        })


# --- KORRIGIERTE DARSTELLUNG DER CHAT-NACHRICHTEN ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Unterscheide, welches Feld angezeigt werden soll, je nach Rolle.
        if msg["role"] == "user":
            # User-Nachrichten haben den Schlüssel "content"
            st.markdown(msg.get("content", ""))
        elif msg["role"] == "bot":
            # Bot-Nachrichten haben den Schlüssel "answer"
            st.markdown(msg.get("answer", ""))

            # Die Anzeige für den Kontext (ehemals "sources") bleibt gleich.
            sources_list = msg.get("sources", [])
            if sources_list:
                with st.expander("Verwendeten Kontext anzeigen"):
                    for src in sources_list:
                        if isinstance(src, dict):
                            content = src.get("content", "Kein Inhalt verfügbar.")
                            st.markdown("---")
                            st.markdown(content)
                        else:
                            st.markdown("---")
                            st.text(str(src))

# Chat-Eingabe-Formular am unteren Rand
st.chat_input("Stelle deine Frage...", key="user_input", on_submit=submit_callback)
