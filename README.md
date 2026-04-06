# Agentic Scraper (ARIA)

A chat-first data extraction assistant that uses search, scraping, and LLM-powered structured extraction.

## 1) Quick setup (copy/paste)

> Run these commands from the project root (`/workspace/agentic_scraper`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m playwright install chromium
```

Create a `.env` file:

```bash
cat > .env <<'ENV'
OPENAI_API_KEY=sk-your-key-here
ENV
```

Run the app:

```bash
python main.py
```

---

## 2) Why you are seeing import errors in VS Code

If lines like these are underlined:

- `from prompt_toolkit import PromptSession`
- `from rich.console import Console`
- `from ddgs import DDGS`

the **most common cause** is that VS Code is not using the same Python interpreter as your venv.

### Fix in VS Code

1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**.
2. Choose the interpreter inside this repo, usually:
   - Linux/macOS: `.venv/bin/python`
   - Windows: `.venv\\Scripts\\python.exe`
3. Open a new terminal in VS Code and run:

```bash
python -V
which python
pip -V
```

(`where python` on Windows.)

The paths should point to your `.venv`.

---

## 3) Verify every problematic import explicitly

Run this one-liner while venv is active:

```bash
python -c "from prompt_toolkit import PromptSession; from prompt_toolkit.history import FileHistory; from prompt_toolkit.auto_suggest import AutoSuggestFromHistory; from prompt_toolkit.styles import Style; from rich.console import Console; from rich.panel import Panel; from rich.rule import Rule; from rich.table import Table; from ddgs import DDGS; print('✅ all imports OK')"
```

If this prints `✅ all imports OK`, your packages are installed correctly.

---

## 4) Runtime dependencies people often miss

Some packages need system binaries too:

- **Playwright browser** (required by the scraper flow):
  - `python -m playwright install chromium`
- **Tesseract OCR** (only needed if OCR paths are used):
  - Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr`
  - macOS (brew): `brew install tesseract`

If Playwright fails on Linux, also run:

```bash
python -m playwright install --with-deps chromium
```

---

## 5) Minimal startup check

```bash
python -m compileall main.py agents tools config.py state.py
python main.py --debug
```

If `.env` is missing, startup will fail fast with a clear `OPENAI_API_KEY` message (expected behavior).

---

## 6) Common fixes if imports still fail

1. **You installed into a different Python**
   - Re-activate venv and reinstall:

   ```bash
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. **Shadowing package names with local files**
   - Ensure you do **not** have files named `rich.py`, `ddgs.py`, or `prompt_toolkit.py` in project root.

3. **Broken venv**
   - Recreate cleanly:

   ```bash
   deactivate || true
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Jupyter / notebook kernel mismatch**
   - If running in notebook, install/select kernel from this exact venv.

---

## 7) Project run modes

- Interactive chat: `python main.py`
- Resume session: `python main.py --resume data/sessions/<session>.json`
- Debug mode: `python main.py --debug`

---

## 8) If you share errors, share these exact outputs

Please paste output of:

```bash
python -V
which python
pip -V
pip show prompt_toolkit rich ddgs
python -c "import prompt_toolkit, rich, ddgs; print(prompt_toolkit.__version__, rich.__version__, ddgs.__version__)"
python main.py --debug
```

With those logs, we can fix remaining issues quickly.
