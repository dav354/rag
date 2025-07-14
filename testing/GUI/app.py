# File: app.py
# Description: v2.0 - Angepasst für die Verarbeitung von Quellen im JSON-Format.

import streamlit as st
import requests
import json # Importieren, um JSON-Strings schön darzustellen

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
                resp = requests.post(API_URL, json={"query": user_input}, timeout=60) # Timeout erhöht
                resp.raise_for_status()
                data = resp.json()

                # Korrektes Extrahieren der Daten aus der neuen API-Antwortstruktur
                answer_data = data.get("answer", {})
                answer_text = answer_data.get("answer", "Keine Antwort erhalten.")
                # Quellen sind jetzt eine Liste von Dictionaries, default ist eine leere Liste
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
            "sources": sources # Speichert die Liste der Quell-Objekte
        })
    # Input-Feld zurücksetzen
    st.session_state.user_input = ""

# --- Darstellung der Chat-Nachrichten ---
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])

    elif msg["role"] == "bot":
        with st.chat_message("bot"):
            st.markdown(msg.get("answer", ""))

            # Neue Sektion für Quellen und Metadaten
            sources_list = msg.get("sources", [])
            if sources_list:
                with st.expander("Quellen und Metadaten anzeigen"):
                    # Anzeige der Dauer direkt hier
                    duration = msg.get("duration", "")
                    if duration:
                        st.caption(f"Antwort generiert in {float(duration):.2f} Sekunden.")

                    # Iteriere durch die JSON-Objekte der Quellen
                    for idx, src in enumerate(sources_list):
                        # Überprüfen, ob die Quelle ein Dictionary ist (neues Format)
                        if isinstance(src, dict):
                            # Extrahiere den Dateipfad aus den Metadaten für die Überschrift
                            file_path = src.get("metadata", {}).get("file_path", f"Quelle {idx + 1}")
                            st.markdown(f"**{file_path}**")

                            # Zeige den Inhalt der Quelle an
                            content = src.get("content", "Kein Inhalt verfügbar.")
                            st.text_area(
                                label=f"Chunk {idx+1}",
                                value=content,
                                height=150,
                                key=f"source_{i}_{idx}",
                                disabled=True
                            )
                        else:
                            # Fallback für altes String-Format (falls noch vorhanden)
                            st.text(str(src))
                    st.divider()


# Chat-Eingabe-Formular am unteren Rand fixieren
st.chat_input("Stelle deine Frage...", key="user_input", on_submit=submit_callback)