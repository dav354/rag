import streamlit as st
import requests
import json
import re

# ✅ MUSS als erste Streamlit-Funktion stehen!
st.set_page_config(page_title="KG-RAG Chatbot", page_icon="🤖")

# Hilfsfunktion zum Formatieren der Links (unverändert)
def format_answer_with_links(answer_text: str) -> str:
    url_pattern = re.compile(r'https?://[^\s()<>]+')
    def replace_link(match):
        url = match.group(0)
        cleaned_url = url.rstrip('/')
        return f"[{cleaned_url}]({cleaned_url})"
    return re.sub(url_pattern, replace_link, answer_text)

def main():
    st.title("askTHWS-ChatBot")

    # ----- EINSTELLUNGEN SEITENLEISTE -----
    st.sidebar.header("Einstellungen")
    if "api_base" not in st.session_state:
        st.session_state.api_base = "http://localhost:8000"
    st.session_state.api_base = st.sidebar.text_input(
        "API Basis-URL", value=st.session_state.api_base
    )
    API_URL = f"{st.session_state.api_base.rstrip('/')}/ask"

    # ----- CHAT-INITIALISIERUNG -----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ----- ANZEIGE DER CHAT-HISTORIE -----
    # Dieser Loop läuft bei jedem Neuladen und zeigt alle bisherigen Nachrichten an.
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "user"):
            # Prüfen, ob es eine Bot- oder User-Nachricht ist
            if "answer" in message: # Bot-Nachricht
                formatted_answer = format_answer_with_links(message.get("answer", ""))
                st.markdown(formatted_answer)
                # Expander für Details, falls vorhanden
                with st.expander("Details und Quellen"):
                    st.text_input("Antwortdauer (s)", message.get("duration", ""), key=f"duration_{message.get('id')}", disabled=True)
                    st.markdown("**Quellen:**")
                    sources = message.get("sources", [])
                    if sources:
                        for src in sources:
                            if src:
                                cleaned_src = str(src).rstrip('/')
                                st.markdown(f"- [{cleaned_src}]({cleaned_src})")
                    else:
                        st.markdown("_Keine Quellen vorhanden._")
            else: # User-Nachricht
                st.markdown(message["content"])


    # ----- CHAT-EINGABE UND VERARBEITUNGSLOGIK -----
    # `st.chat_input` wartet auf eine neue Nutzereingabe.
    if prompt := st.chat_input("Schreibe hier deine Frage..."):
        # 1. Füge die User-Nachricht zur Historie hinzu und zeige sie sofort an.
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Erhalte und verarbeite die Bot-Antwort.
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Bot denkt nach…"):
                try:
                    resp = requests.post(API_URL, json={"query": prompt}, timeout=10000)
                    resp.raise_for_status()
                    data = resp.json()

                    # Bot-Antwort aus der API extrahieren
                    answer_text = data.get("answer", {}).get("answer", "Keine Antwort erhalten.")
                    duration = str(data.get("duration_seconds", ""))
                    sources_raw = data.get("answer", {}).get("sources", "")

                    # Quellen-Verarbeitung
                    if isinstance(sources_raw, str):
                        try:
                            sources_list = json.loads(sources_raw) if sources_raw else []
                        except json.JSONDecodeError:
                            sources_list = [sources_raw] if sources_raw else []
                    else:
                        sources_list = sources_raw if isinstance(sources_raw, list) else []

                    # 3. Füge die Bot-Nachricht zur Historie hinzu.
                    bot_message = {
                        "role": "assistant",
                        "answer": answer_text,
                        "duration": duration,
                        "sources": sources_list,
                        "id": len(st.session_state.messages) # Eindeutiger Key für Widgets
                    }
                    st.session_state.messages.append(bot_message)

                    # 4. Zeige die neue Bot-Antwort sofort an.
                    formatted_answer = format_answer_with_links(answer_text)
                    st.markdown(formatted_answer)
                    with st.expander("Details und Quellen"):
                        st.text_input("Antwortdauer (s)", duration, key=f"duration_{bot_message['id']}", disabled=True)
                        st.markdown("**Quellen:**")
                        if sources_list:
                            for src in sources_list:
                                if src:
                                    cleaned_src = str(src).rstrip('/')
                                    st.markdown(f"- [{cleaned_src}]({cleaned_src})")
                        else:
                            st.markdown("_Keine Quellen vorhanden._")

                except Exception as e:
                    error_message = f"Fehler bei der Anfrage an die API: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "answer": error_message, "id": len(st.session_state.messages)})


if __name__ == "__main__":
    main()