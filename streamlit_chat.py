"""
Streamlit Chat Interface for Epstein Docs Agent API
"""

import os
import re
import streamlit as st
import requests
import json
import uuid


def fetch_document(api_url: str, api_key: str, doc_id: str) -> dict | None:
    """Fetch a document from the API."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        response = requests.get(f"{api_url}/api/documents/{doc_id}", headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def render_text_with_sources(text: str, sources_dict: dict, api_url: str, api_key: str, msg_idx: int):
    """
    Render text with inline clickable doc_id references.

    Detects doc_ids in the text and makes them clickable - clicking opens
    a popover with the full document content.

    Args:
        text: The response text from the agent
        sources_dict: Dict mapping doc_id to source metadata/content
        api_url: API URL for fetching full documents
        api_key: API key for authentication
        msg_idx: Message index for unique keys
    """
    if not sources_dict:
        st.markdown(text)
        return

    doc_ids = list(sources_dict.keys())
    if not doc_ids:
        st.markdown(text)
        return

    # Escape special regex characters and build pattern
    escaped_ids = [re.escape(doc_id) for doc_id in doc_ids]
    pattern = r'(' + '|'.join(escaped_ids) + r')'

    # Split text by doc_id matches
    parts = re.split(pattern, text)

    # Render parts
    i = 0
    for part in parts:
        if part in sources_dict:
            # This is a doc_id - render as clickable popover
            source_info = sources_dict.get(part, {})
            with st.popover(f"📄 {part}"):
                st.markdown(f"### {part}")

                # Show metadata if available
                if isinstance(source_info, dict):
                    if source_info.get("category"):
                        st.markdown(f"**Catégorie:** {source_info['category']}")
                    if source_info.get("date_range"):
                        st.markdown(f"**Période:** {source_info['date_range']}")
                    if source_info.get("summary"):
                        st.caption(source_info["summary"])

                st.divider()

                # Load and show full text directly
                doc_data = fetch_document(api_url, api_key, part)
                if doc_data and doc_data.get("full_text"):
                    st.text_area(
                        "Contenu complet",
                        value=doc_data["full_text"],
                        height=400,
                        key=f"inline_text_{msg_idx}_{i}_{part}"
                    )
                else:
                    st.warning("Document non trouvé")
            i += 1
        else:
            # Regular text - render as markdown
            if part:
                st.markdown(part, unsafe_allow_html=True)


def display_source_card(source: dict, api_url: str, api_key: str, idx: int):
    """Display a source card with metadata and view button."""
    doc_id = source.get("doc_id") if isinstance(source, dict) else source
    summary = source.get("summary", "") if isinstance(source, dict) else ""
    category = source.get("category", "") if isinstance(source, dict) else ""
    date_range = source.get("date_range", "") if isinstance(source, dict) else ""
    score = source.get("score") if isinstance(source, dict) else None

    # Container for the source card
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            # Document ID as header
            st.markdown(f"**{doc_id}**")

            # Metadata pills
            meta_parts = []
            if category:
                meta_parts.append(f"`{category}`")
            if date_range and date_range != "? - ?":
                meta_parts.append(f"📅 {date_range}")
            if score is not None:
                meta_parts.append(f"🎯 {score:.2f}")

            if meta_parts:
                st.markdown(" • ".join(meta_parts))

            # Summary
            if summary:
                st.caption(summary[:200] + ("..." if len(summary) > 200 else ""))

        with col2:
            # View document button
            if st.button("📖 Voir", key=f"view_doc_{idx}_{doc_id}"):
                st.session_state[f"show_doc_{doc_id}"] = True

        # Show document content if button was clicked
        if st.session_state.get(f"show_doc_{doc_id}", False):
            with st.expander(f"📄 Contenu de {doc_id}", expanded=True):
                doc_data = fetch_document(api_url, api_key, doc_id)
                if doc_data:
                    # Document metadata
                    st.markdown(f"**Catégorie:** {doc_data.get('category', 'N/A')}")
                    st.markdown(f"**Période:** {doc_data.get('date_range', 'N/A')}")
                    if doc_data.get("paragraph_summary"):
                        st.info(doc_data["paragraph_summary"])

                    st.divider()

                    # Full text
                    full_text = doc_data.get("full_text", "")
                    if full_text:
                        st.text_area(
                            "Texte complet",
                            value=full_text,
                            height=400,
                            key=f"text_area_{doc_id}"
                        )
                    else:
                        st.warning("Texte non disponible")
                else:
                    st.error(f"Impossible de charger le document {doc_id}")

                # Close button
                if st.button("Fermer", key=f"close_doc_{doc_id}"):
                    st.session_state[f"show_doc_{doc_id}"] = False
                    st.rerun()

        st.divider()

# Default values from environment
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:3002")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")

# Page config
st.set_page_config(
    page_title="Epstein Docs Agent",
    page_icon="🔍",
    layout="centered"
)

st.title("Epstein Docs Agent Chat")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="URL de l'API (local ou production)"
    )
    st.session_state.api_url = api_url

    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", DEFAULT_API_KEY),
        type="password",
        help="Clé API pour l'authentification"
    )
    st.session_state.api_key = api_key

    st.divider()

    # Session management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.text(f"Session: {st.session_state.session_id[:8]}...")

    if st.button("Nouvelle conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Test connection
    if st.button("Tester la connexion"):
        try:
            response = requests.get(f"{api_url}/", timeout=5)
            if response.status_code == 200:
                st.success("Connexion OK")
                stats_response = requests.get(f"{api_url}/api/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    st.info(f"Documents: {stats.get('total_documents', 'N/A')}")
            else:
                st.error(f"Erreur: {response.status_code}")
        except Exception as e:
            st.error(f"Connexion échouée: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # For assistant messages with sources, render with inline clickable references
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            # Build sources dict for inline rendering
            sources_dict = {}
            for source in message["sources"]:
                if isinstance(source, dict):
                    doc_id = source.get("doc_id")
                    if doc_id:
                        sources_dict[doc_id] = source
                else:
                    # Legacy format: source is just a string doc_id
                    sources_dict[source] = {"doc_id": source}

            # Render text with inline clickable sources
            render_text_with_sources(
                message["content"],
                sources_dict,
                st.session_state.api_url,
                st.session_state.api_key,
                msg_idx
            )
        else:
            st.markdown(message["content"])

        # Show tools used
        if message["role"] == "assistant" and "tools" in message and message["tools"]:
            with st.expander(f"🔧 Outils utilisés ({len(message['tools'])})"):
                for tool in message["tools"]:
                    st.markdown(f"**{tool['name']}**")
                    st.code(json.dumps(tool["args"], indent=2, ensure_ascii=False), language="json")

        # Show all sources in collapsible section (still keep this for easy browsing)
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(f"📄 Toutes les sources ({len(message['sources'])})"):
                for idx, source in enumerate(message["sources"]):
                    display_source_card(
                        source,
                        st.session_state.api_url,
                        st.session_state.api_key,
                        idx=f"history_{msg_idx}_{idx}"
                    )

# Chat input
if prompt := st.chat_input("Posez votre question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request
    headers = {"Content-Type": "application/json"}
    if st.session_state.api_key:
        headers["X-API-Key"] = st.session_state.api_key

    payload = {
        "question": prompt,
        "session_id": st.session_state.session_id
    }

    # Get streaming response
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{st.session_state.api_url}/api/query/stream",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"Erreur API: {response.status_code} - {response.text}")
            else:
                # Containers for dynamic content
                tool_container = st.container()
                message_placeholder = st.empty()

                full_response = ""
                all_sources = []
                all_tools = []
                current_tool = None

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            try:
                                event = json.loads(line_str[6:])
                                event_type = event.get("type", "")

                                if event_type == "token":
                                    content = event.get("content", "")
                                    full_response += content
                                    message_placeholder.markdown(full_response + "▌")

                                elif event_type == "tool_start":
                                    tool_name = event.get("tool_name", "unknown")
                                    tool_args = event.get("tool_args", {})
                                    current_tool = {"name": tool_name, "args": tool_args}
                                    all_tools.append(current_tool)

                                    # Display tool being called
                                    with tool_container:
                                        st.info(f"🔧 Appel: **{tool_name}**")

                                elif event_type in ("tool_end", "graph"):
                                    tool_name = event.get("tool_name", "unknown")
                                    sources = event.get("sources", [])

                                    # Collect sources (now with metadata)
                                    for src in sources:
                                        # Check if source already exists by doc_id
                                        src_id = src.get("doc_id") if isinstance(src, dict) else src
                                        existing_ids = [
                                            s.get("doc_id") if isinstance(s, dict) else s
                                            for s in all_sources
                                        ]
                                        if src_id not in existing_ids:
                                            all_sources.append(src)

                                    with tool_container:
                                        if event_type == "graph":
                                            graph = event.get("graph", {})
                                            n_nodes = len(graph.get("nodes", []))
                                            n_edges = len(graph.get("edges", []))
                                            st.success(f"✓ {tool_name} - {n_nodes} personnes, {n_edges} liens, {len(sources)} doc(s)")
                                        elif sources:
                                            st.success(f"✓ {tool_name} - {len(sources)} doc(s) trouvé(s)")
                                        else:
                                            st.success(f"✓ {tool_name}")

                                elif event_type == "error":
                                    st.error(f"Erreur: {event.get('content', 'Unknown error')}")
                                    break

                            except json.JSONDecodeError:
                                continue

                # Final display
                message_placeholder.markdown(full_response)

                # Show tools summary
                if all_tools:
                    with st.expander(f"🔧 Outils utilisés ({len(all_tools)})"):
                        for tool in all_tools:
                            st.markdown(f"**{tool['name']}**")
                            st.code(json.dumps(tool["args"], indent=2, ensure_ascii=False), language="json")

                # Show sources summary with metadata and view buttons
                if all_sources:
                    with st.expander(f"📄 Sources ({len(all_sources)})"):
                        for idx, source in enumerate(all_sources):
                            display_source_card(
                                source,
                                st.session_state.api_url,
                                st.session_state.api_key,
                                idx=f"stream_{idx}"
                            )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": all_sources,
                    "tools": all_tools
                })

                # Rerun to display with clickable sources
                if all_sources:
                    st.rerun()

        except requests.exceptions.Timeout:
            st.error("Timeout - la requête a pris trop de temps")
        except Exception as e:
            st.error(f"Erreur: {e}")
