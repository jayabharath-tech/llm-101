# Build LLM 101 Assignment

- Install ollama on your local machine and host llm models.
- Use openai python library to connect to ollama and generate responses.
- Streamlit library to create a chat interface.
  - Includes the moderation check. Ignores the flagged messages from the context.
  - Maintains the context/conversations.

## Developer tools

### ollama

```bash
    brew install ollama
    ollama serve
    ollama --version
    ollama pull llama3.1
    ollama list
```

ollama runs on port 11434 by default

### Streamlit

```bash
    pip install streamlit
    streamlit --version
    streamlit run story_basic.py
```
