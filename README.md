# Gemma 4 RAG TUI

A terminal-based Knowledge Base & RAG interface powered by **Gemma 4** (`gemma-4-31b-it`), **LangChain**, **ChromaDB**, and **Textual**.

## Features

- **Terminal User Interface (TUI):** Built with Textual for a terminal experience with keyboard shortcuts.
- **Multi-Format Document Ingestion:** Supports `.pdf`, `.docx`, `.csv`, `.txt`, `.md`, `.html`, and code/script files.
- **Tabular Schema Awareness:** Injects column headers into row chunks for tabular analysis and comparative queries.
- **Smart Vector Cache:** MD5 fingerprinting re-uses ChromaDB vector embeddings across launches if files in `docs/` haven't changed.
- **No-Docs Fallback:** Boots into direct Chat Mode if `docs/` is empty without crashing.
- **Slash Commands & Shortcuts:** Built-in commands (`/docs`, `/reload`, `/clear`, `/sources`, `/model`, `/export`) and hotkeys (`Ctrl+L`, `Ctrl+R`, `Ctrl+S`).
- **Deterministic Extraction:** Configured with `temperature=0.0` and `max_tokens=2048` to eliminate hallucination loops and mid-sentence cutoffs.

---

## Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/gemma-rag-project.git
cd gemma-rag-project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey):

```bash
cp .env.example .env
# Edit .env and paste your API key:
# GOOGLE_API_KEY="AIzaSy..."
```

*(Alternatively, you can just start the app — it will securely prompt you for your API key on first launch).*

### 3. Add Documents & Run

```bash
# Add your documents (PDFs, CSVs, etc.)
cp /path/to/my_data.pdf docs/

# Launch the TUI
python rag_app.py
```

---

## Commands & Keyboard Shortcuts

| Command | Keybinding | Action |
|---|---|---|
| `/help` | — | Display all available commands |
| `/docs` | — | List all indexed files, types, and chunk counts |
| `/reload` | `Ctrl+R` | Re-index `docs/` on the fly without restarting |
| `/clear` | `Ctrl+L` | Clear chat window and conversation memory |
| `/sources` | — | Toggle source citations on/off |
| `/model` | — | View active model, provider, temperature, and mode |
| `/export` | `Ctrl+S` | Export chat history to timestamped Markdown (`.md`) |
| — | `d` | Toggle dark/light theme |
| — | `q` | Quit the application |

---

## Standalone Binary Packaging

To bundle the entire application into a standalone portable binary:

```bash
chmod +x build_dist.sh
./build_dist.sh
```

Outputs will be placed in `dist/gemma-tui/` and packaged into `dist/gemma-tui-linux.tar.gz`.
