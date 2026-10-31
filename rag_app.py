import os
import re
import warnings
import multiprocessing
import hashlib
import time
import shutil
import csv
from pathlib import Path
from datetime import datetime

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

warnings.filterwarnings("ignore")

from dotenv import load_dotenv, set_key
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual import work
from rich.markdown import Markdown

load_dotenv()

# ---------------------------------------------------------------------------
# Supported Document Types & Ingestion
# ---------------------------------------------------------------------------

FINGERPRINT_FILE = ".docs_fingerprint"

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf": "PDF",
    ".docx": "Word Document",
    ".txt": "Text File",
    ".md": "Markdown",
    ".markdown": "Markdown",
    # Data & Web
    ".csv": "CSV Spreadsheet",
    ".json": "JSON Data",
    ".html": "HTML Document",
    ".htm": "HTML Document",
    # Code & Config
    ".py": "Python Script",
    ".c": "C Source",
    ".cpp": "C++ Source",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".rs": "Rust Source",
    ".go": "Go Source",
    ".java": "Java Source",
    ".sh": "Shell Script",
    ".sql": "SQL Query",
    ".yaml": "YAML Config",
    ".yml": "YAML Config",
}


def compute_docs_fingerprint(docs_dir="docs"):
    """Hash all supported file names, sizes, and modification times."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return None
    entries = []
    for f in sorted(docs_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            st = f.stat()
            entries.append(f"{f.name}:{st.st_size}:{st.st_mtime}")
    return hashlib.md5("|".join(entries).encode()).hexdigest() if entries else None


def load_saved_fingerprint():
    try:
        return Path(FINGERPRINT_FILE).read_text().strip()
    except FileNotFoundError:
        return None


def save_fingerprint(fp):
    Path(FINGERPRINT_FILE).write_text(fp)


def load_csv_document(file_path: Path) -> list[Document]:
    """
    Load CSV files injecting header schema metadata into each row chunk.
    Handles standard CSVs and CSVs with quote-wrapped lines.
    Ensures every row maintains full schema awareness:
    e.g., 'Asset: ..., Date: ..., Gamma: ..., Slippage: ...'
    """
    docs = []
    try:
        import io
        raw_text = file_path.read_text(encoding="utf-8-sig", errors="ignore")
        cleaned_lines = []
        for line in raw_text.splitlines():
            line_str = line.strip()
            if line_str.startswith('"') and line_str.endswith('"') and line_str.count('"') >= 2:
                # Remove surrounding line quotes if present
                line_str = line_str[1:-1]
            cleaned_lines.append(line_str)

        reader = csv.DictReader(io.StringIO("\n".join(cleaned_lines)))
        fieldnames = [fn.strip() for fn in (reader.fieldnames or []) if fn]
        header_str = ", ".join(fieldnames)

        for row_idx, row in enumerate(reader):
            pairs = [
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                if k and v is not None and v.strip() != ""
            ]
            if not pairs:
                continue
            schema_row = ", ".join(pairs)
            content = f"[Table: {file_path.name} | Row {row_idx + 1}]\n{schema_row}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "row": row_idx + 1,
                        "type": "tabular",
                        "headers": header_str,
                    },
                )
            )
    except Exception:
        try:
            return CSVLoader(str(file_path), encoding="utf-8").load()
        except Exception:
            return []
    return docs


def load_document_file(file_path: Path) -> list[Document]:
    """Load a single non-tabular document based on its extension."""
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(str(file_path)).load()
        elif ext == ".docx":
            return Docx2txtLoader(str(file_path)).load()
        elif ext in [".html", ".htm"]:
            return BSHTMLLoader(str(file_path), open_encoding="utf-8").load()
        elif ext in SUPPORTED_EXTENSIONS:
            return TextLoader(str(file_path), encoding="utf-8").load()
    except Exception:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                return [Document(page_content=content, metadata={"source": str(file_path)})]
        except Exception:
            return []
    return []


def load_all_documents(docs_dir="docs") -> tuple[list[Document], list[Document]]:
    """
    Scan and load all supported documents.
    Returns: (text_docs, tabular_docs)
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return [], []
    text_docs = []
    tabular_docs = []
    for f in sorted(docs_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            if f.suffix.lower() == ".csv":
                tabular_docs.extend(load_csv_document(f))
            else:
                text_docs.extend(load_document_file(f))
    return text_docs, tabular_docs


# ---------------------------------------------------------------------------
# Output Cleaner (Strips <thought> tags and meta-scratchpads)
# ---------------------------------------------------------------------------

def clean_model_output(text: str) -> str:
    """
    Strips raw <thought>...</thought> tags, incomplete thought blocks,
    and meta scratchpad notes from model answers.
    """
    text = text.strip()
    if not text:
        return ""

    # Case 1: Complete thought block present
    if "<thought>" in text and "</thought>" in text:
        parts = text.split("</thought>", 1)
        after_thought = parts[1].strip()
        if after_thought:
            text = after_thought
        else:
            text = parts[0].replace("<thought>", "").strip()
    elif text.startswith("<thought>"):
        text = text.replace("<thought>", "", 1).strip()
    elif "</thought>" in text:
        text = text.replace("</thought>", "").strip()

    # Case 2: Strip meta scratchpad commentary
    lines = text.splitlines()
    cleaned_lines = []
    skipping_meta = True
    meta_pattern = re.compile(
        r"^\s*([*\-•]\s*)?(task|constraint|instructions|thinking process|note|scratchpad)\s*:",
        re.IGNORECASE,
    )
    for line in lines:
        if skipping_meta:
            if not line.strip() or meta_pattern.match(line):
                continue
            skipping_meta = False
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result if result else text


# ---------------------------------------------------------------------------
# Screen: API Key Entry
# ---------------------------------------------------------------------------

class ApiKeyScreen(Screen):
    """Startup screen to securely input and save the Google AI Studio API Key."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            yield Static(
                "\n\n[bold cyan]Welcome to the DevDrafts RAG Pipeline[/bold cyan]\n",
                id="title-text",
            )
            yield Static("Please enter your Google AI Studio API Key to continue:")
            yield Input(placeholder="Paste your API key here...", password=True, id="api-key-input")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        key = event.value.strip()
        if len(key) >= 20:
            set_key(".env", "GOOGLE_API_KEY", key)
            os.environ["GOOGLE_API_KEY"] = key
            self.app.switch_screen("main")
        else:
            self.app.notify("Invalid key — must be at least 20 characters.", severity="error")


# ---------------------------------------------------------------------------
# Screen: Main Chat Interface
# ---------------------------------------------------------------------------

class MainScreen(Screen):
    """The main RAG Chat Interface with deterministic tabular extraction & thread-safe rendering."""

    BINDINGS = [
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+r", "reload_docs", "Reload"),
        ("ctrl+s", "export_chat", "Export"),
    ]

    def __init__(self):
        super().__init__()
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        self.chat_history: list[tuple[str, str]] = []
        self.conversation_log: list[tuple[str, str]] = []
        self.show_sources = True
        self.chunk_count = 0
        self.mode = "rag"  # "rag" | "chat"

    # ---- Layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            RichLog(id="chat-log", highlight=True, markup=True),
            Horizontal(
                Input(
                    placeholder="Ask a question… (type /help for commands)",
                    id="query-input",
                ),
                id="input-container",
            ),
            id="chat-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Header).title = "Gemma 4 TUI Knowledge Base"
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("[bold yellow]⚡ Initializing RAG Knowledge Base…[/bold yellow]")
        self.init_rag()

    # ---- RAG Initialization ----------------------------------------------

    @work(thread=True)
    def init_rag(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        try:
            start = time.time()

            # 1. LLM Client: temperature=0.0 for deterministic extraction, max_tokens=2048 to prevent truncation
            self.llm = ChatOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.environ.get("GOOGLE_API_KEY"),
                model="gemma-4-31b-it",
                temperature=0.0,
                max_tokens=2048,
                max_retries=3,
            )

            # 2. Check documents
            current_fp = compute_docs_fingerprint()

            if current_fp is None:
                self.mode = "chat"
                self.rag_chain = None
                elapsed = time.time() - start
                self.app.call_from_thread(
                    chat_log.write,
                    f"[bold green]✅ Chat Mode (no docs). Gemma 4 online. ({elapsed:.1f}s)[/bold green]",
                )
                self.app.call_from_thread(
                    chat_log.write,
                    "[dim]Add files to docs/ and type /reload to enable RAG mode.[/dim]\n",
                )
                return

            # 3. Smart caching — only rebuild when documents change
            saved_fp = load_saved_fingerprint()
            chroma_exists = Path("./chroma_db").exists()
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            if chroma_exists and saved_fp == current_fp:
                self.app.call_from_thread(
                    chat_log.write, "[dim]Using cached vector database…[/dim]"
                )
                vector_db = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embeddings,
                )
                self.chunk_count = vector_db._collection.count()
            else:
                self.app.call_from_thread(
                    chat_log.write,
                    "[dim]Building vector database (first run or docs changed)…[/dim]",
                )
                text_docs, tabular_docs = load_all_documents("docs")

                if not text_docs and not tabular_docs:
                    self.mode = "chat"
                    self.rag_chain = None
                    self.app.call_from_thread(
                        chat_log.write,
                        "[yellow]⚠️ No readable content found in docs/. Running in Chat Mode.[/yellow]\n",
                    )
                    return

                # For PDFs and DOCX, split with chunk_size=1200 and chunk_overlap=250
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=250,
                )
                text_chunks = text_splitter.split_documents(text_docs) if text_docs else []

                # Tabular rows already contain atomic header schema; combine directly
                chunks = text_chunks + tabular_docs
                self.chunk_count = len(chunks)

                if chroma_exists:
                    shutil.rmtree("./chroma_db")

                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db",
                )
                save_fingerprint(current_fp)

            # 4. Retriever depth increased to k=5 for multi-row tabular context
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 5})

            # 5. RAG System Prompt enforcing tabular precision
            rag_system_prompt = (
                "You are a technical risk and execution assistant. Answer strictly based on the provided context. "
                "When analyzing tables, compare all rows before identifying maximums, minimums, or violations. "
                "If numerical data is ambiguous or missing, state so explicitly. Format responses cleanly in markdown.\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", rag_system_prompt),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
            self.mode = "rag"
            elapsed = time.time() - start

            self.app.call_from_thread(
                chat_log.write,
                f"[bold green]✅ RAG Ready. {self.chunk_count} chunks indexed (k=5). "
                f"Gemma 4 online. ({elapsed:.1f}s)[/bold green]\n",
            )

        except Exception as e:
            self.app.call_from_thread(
                chat_log.write, f"[bold red]Initialization Error:[/bold red] {e}"
            )

    # ---- Input Handling --------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "query-input":
            return
        if not self.llm:
            self.app.notify("Still initializing…", severity="warning")
            return

        query = event.value.strip()
        if not query:
            return
        event.input.value = ""

        chat_log = self.query_one("#chat-log", RichLog)

        # Slash commands
        if query.startswith("/"):
            self._handle_slash_command(query, chat_log)
            return

        # Guard: RAG mode requires initialized chain
        if self.mode == "rag" and self.rag_chain is None:
            self.app.notify("Documents still loading…", severity="warning")
            return

        chat_log.write(f"\n[bold blue]You:[/bold blue] {query}")
        chat_log.write("[italic grey]Analyzing context & generating response…[/italic grey]")
        self.conversation_log.append(("user", query))
        self.fetch_answer(query)

    # ---- Thread-Safe Answer Execution ------------------------------------

    @work(thread=True)
    def fetch_answer(self, query: str) -> None:
        start = time.time()
        try:
            if self.mode == "rag" and self.rag_chain:
                # Execute full retrieval chain safely
                response = self.rag_chain.invoke({"input": query})
                raw_answer = response.get("answer", "")
                source_docs = response.get("context", [])
            else:
                # Chat mode fallback
                system_prompt = (
                    "You are a technical risk and execution assistant. "
                    "Answer accurately, concisely, and format cleanly in markdown."
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query),
                ]
                response = self.llm.invoke(messages)
                raw_answer = response.content
                source_docs = []

            # Clean output of any thoughts or scratchpads
            clean_answer = clean_model_output(raw_answer)
            if not clean_answer:
                clean_answer = raw_answer.strip()

            elapsed = time.time() - start

            # Safely dispatch complete UI rendering to the main thread
            self.app.call_from_thread(
                self._render_chat_response, clean_answer, source_docs, elapsed
            )

            # Update memory & export log
            self.chat_history.append(("user", query))
            self.chat_history.append(("assistant", clean_answer))
            self.conversation_log.append(("assistant", clean_answer))

        except Exception as e:
            err_msg = str(e)
            self.app.call_from_thread(self._render_error, err_msg)

    # ---- Main-Thread Safe UI Renderers -----------------------------------

    def _render_chat_response(self, answer: str, source_docs: list, elapsed: float) -> None:
        """Executed on the main UI event loop to safely write complete markdown without truncation."""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("\n[bold cyan]Gemma:[/bold cyan]")
        chat_log.write(Markdown(answer))

        if self.show_sources and source_docs:
            chat_log.write("[dim]Sources:[/dim]")
            seen = set()
            for doc in source_docs:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page")
                row = doc.metadata.get("row")
                key = f"{src}:{page}:{row}"
                if key not in seen:
                    seen.add(key)
                    fname = Path(src).name
                    if page is not None:
                        page_display = int(page) + 1 if isinstance(page, (int, float)) else page
                        chat_log.write(f"  [dim]📄 {fname} — Page {page_display}[/dim]")
                    elif row is not None:
                        chat_log.write(f"  [dim]📊 {fname} — Row {row}[/dim]")
                    else:
                        chat_log.write(f"  [dim]📄 {fname}[/dim]")

        chat_log.write(f"[dim]({elapsed:.1f}s)[/dim]")
        chat_log.write("─" * 50)

    def _render_error(self, err_msg: str) -> None:
        """Executed on the main UI event loop to display formatted errors."""
        chat_log = self.query_one("#chat-log", RichLog)
        if "429" in err_msg:
            chat_log.write("[bold red]Rate limited — wait a moment and try again.[/bold red]")
        elif "401" in err_msg or "403" in err_msg:
            chat_log.write("[bold red]Auth error — check your API key.[/bold red]")
        else:
            chat_log.write(f"[bold red]API Error:[/bold red] {err_msg}")

    # ---- Slash Commands --------------------------------------------------

    def _handle_slash_command(self, command: str, chat_log: RichLog) -> None:
        cmd = command.strip().lower()

        if cmd == "/help":
            chat_log.write("\n[bold cyan]━━ Commands ━━[/bold cyan]")
            chat_log.write("  /help       Show this help")
            chat_log.write("  /docs       List loaded documents")
            chat_log.write("  /reload     Re-ingest documents from docs/")
            chat_log.write("  /clear      Clear chat & memory")
            chat_log.write("  /sources    Toggle source citations")
            chat_log.write("  /model      Show model & mode info")
            chat_log.write("  /export     Save conversation to markdown")
            chat_log.write("")
            chat_log.write("[bold cyan]━━ Shortcuts ━━[/bold cyan]")
            chat_log.write("  Ctrl+L      Clear chat")
            chat_log.write("  Ctrl+R      Reload documents")
            chat_log.write("  Ctrl+S      Export conversation")
            chat_log.write("  d           Toggle dark mode")
            chat_log.write("  q           Quit")
            chat_log.write("")

        elif cmd == "/docs":
            docs_path = Path("docs")
            files = (
                [
                    f
                    for f in sorted(docs_path.iterdir())
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
                if docs_path.exists()
                else []
            )
            if files:
                chat_log.write(
                    f"\n[bold cyan]📂 Documents ({len(files)} files, "
                    f"{self.chunk_count} chunks):[/bold cyan]"
                )
                for f in files:
                    size_kb = f.stat().st_size / 1024
                    ext = f.suffix.lower()
                    type_name = SUPPORTED_EXTENSIONS.get(ext, "File")
                    icon = (
                        "📑" if ext == ".pdf"
                        else "📝" if ext == ".docx"
                        else "📊" if ext in [".csv", ".json", ".yaml", ".yml"]
                        else "💻" if ext in [".py", ".c", ".cpp", ".js", ".ts", ".rs", ".go", ".sh", ".sql"]
                        else "📄"
                    )
                    chat_log.write(
                        f"  {icon} [bold]{f.name}[/bold] [dim]({type_name}, {size_kb:.1f} KB)[/dim]"
                    )
            else:
                chat_log.write("\n[yellow]No supported files in docs/ folder.[/yellow]")
            chat_log.write("")

        elif cmd == "/reload":
            chat_log.write("\n[bold yellow]🔄 Reloading documents…[/bold yellow]")
            self.retriever = None
            self.rag_chain = None
            if Path("./chroma_db").exists():
                shutil.rmtree("./chroma_db")
            fp = Path(FINGERPRINT_FILE)
            if fp.exists():
                fp.unlink()
            self.chat_history.clear()
            self.init_rag()

        elif cmd == "/clear":
            self.chat_history.clear()
            self.conversation_log.clear()
            chat_log.clear()
            chat_log.write("[dim]Chat and memory cleared.[/dim]\n")

        elif cmd == "/sources":
            self.show_sources = not self.show_sources
            state = "ON ✅" if self.show_sources else "OFF"
            chat_log.write(f"\n[cyan]Source citations: {state}[/cyan]\n")

        elif cmd == "/model":
            chat_log.write("\n[bold cyan]━━ Status ━━[/bold cyan]")
            chat_log.write("  Model       gemma-4-31b-it")
            chat_log.write("  Provider    Google AI Studio")
            chat_log.write("  Temp        0.0 (Deterministic)")
            chat_log.write("  Max Tokens  2048")
            chat_log.write(f"  Mode        {self.mode.upper()}")
            chat_log.write(f"  Chunks      {self.chunk_count}")
            chat_log.write(f"  Depth (k)   5")
            chat_log.write(f"  Memory      {len(self.chat_history) // 2} exchanges")
            chat_log.write(f"  Sources     {'ON' if self.show_sources else 'OFF'}")
            chat_log.write("")

        elif cmd == "/export":
            self._export_conversation(chat_log)

        else:
            chat_log.write(f"[yellow]Unknown command: {cmd}  (type /help)[/yellow]\n")

    def _export_conversation(self, chat_log: RichLog) -> None:
        if not self.conversation_log:
            chat_log.write("\n[yellow]Nothing to export.[/yellow]\n")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{ts}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# DevDrafts RAG Conversation\n")
            f.write(f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n")
            for role, content in self.conversation_log:
                if role == "user":
                    f.write(f"**You:** {content}\n\n")
                else:
                    f.write(f"**Gemma:**\n\n{content}\n\n---\n\n")
        chat_log.write(f"\n[green]💾 Saved to {filename}[/green]\n")

    # ---- Keyboard shortcut actions ---------------------------------------

    def action_clear_chat(self) -> None:
        self._handle_slash_command("/clear", self.query_one("#chat-log", RichLog))

    def action_reload_docs(self) -> None:
        self._handle_slash_command("/reload", self.query_one("#chat-log", RichLog))

    def action_export_chat(self) -> None:
        self._handle_slash_command("/export", self.query_one("#chat-log", RichLog))


# ---------------------------------------------------------------------------
# App Controller
# ---------------------------------------------------------------------------

class RagTUI(App):
    """Main App Controller."""

    CSS = """
    Screen { background: $surface; }
    #chat-container { height: 1fr; }
    #chat-log { height: 1fr; border: solid $accent; padding: 1; }
    #input-container { height: 3; dock: bottom; margin-bottom: 1; }
    #query-input { width: 1fr; }
    #api-key-input { width: 60; margin-top: 1; }
    Center { align: center middle; }
    #title-text { text-align: center; }
    """

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit", "Quit")]
    SCREENS = {"key": ApiKeyScreen, "main": MainScreen}

    def on_mount(self) -> None:
        """Route to the correct screen based on the .env file."""
        if os.environ.get("GOOGLE_API_KEY"):
            self.push_screen("main")
        else:
            self.push_screen("key")


if __name__ == "__main__":
    app = RagTUI()
    app.run()
