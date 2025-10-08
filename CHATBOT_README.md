# 🤖 AI Multilingual Chatbot

An intelligent chatbot that can respond in multiple languages and answer various questions.

## 🌟 Features

- **Multilingual Support**: Responds in 15+ languages including English, Spanish, French, German, Hindi, Chinese, Japanese, Arabic, and more
- **Multiple AI Backends**: 
  - Fallback mode (no setup required)
  - Ollama integration (local LLM)
  - HuggingFace API integration
- **Web Interface**: Beautiful Streamlit-based UI
- **Command-Line Interface**: Simple terminal-based chatbot
- **Conversation History**: Saves chat history
- **Real-time Responses**: Fast and interactive

## 📋 Supported Languages

1. English
2. Spanish (Español)
3. French (Français)
4. German (Deutsch)
5. Italian (Italiano)
6. Portuguese (Português)
7. Russian (Русский)
8. Chinese (中文)
9. Japanese (日本語)
10. Korean (한국어)
11. Arabic (العربية)
12. Hindi (हिंदी)
13. Bengali (বাংলা)
14. Tamil (தமிழ்)
15. Telugu (తెలుగు)
16. Marathi (मराठी)

## 🚀 Quick Start

### Option 1: Web Interface (Streamlit)

```bash
# Activate environment
conda activate cmrhack

# Run the chatbot
streamlit run app/ai_chatbot.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Command-Line Interface

```bash
# Activate environment
conda activate cmrhack

# Run CLI chatbot
python app/chatbot_cli.py
```

## 🔧 Setup Options

### 1. Fallback Mode (Default - No Setup Required)

The chatbot works out of the box with built-in responses for:
- Greetings
- Common questions
- Basic calculations
- General knowledge
- Time/date queries

**No additional setup needed!**

### 2. Ollama Integration (Local LLM)

For advanced AI responses using local models:

```bash
# Install Ollama
# Download from: https://ollama.ai

# Pull a model
ollama pull llama2

# Start Ollama (it runs automatically after installation)
# The chatbot will automatically detect it
```

**Supported Models:**
- `llama2` (recommended)
- `mistral`
- `codellama`
- `neural-chat`

### 3. HuggingFace API Integration

For cloud-based AI responses:

```bash
# Get API key from https://huggingface.co/settings/tokens

# Set environment variable (Windows PowerShell)
$env:HF_API_KEY="your_api_key_here"

# Or set permanently in System Environment Variables
```

## 💡 Usage Examples

### Web Interface

1. **Select Language**: Choose your preferred response language from the sidebar
2. **Select AI Mode**: Choose between Fallback, Ollama, or HuggingFace
3. **Start Chatting**: Type your message and press Send

### Command-Line Interface

```
You: Hello!
🤖 Bot: Hello! How can I help you today?

You: What is machine learning?
🤖 Bot: Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.

You: Calculate 25 + 37
🤖 Bot: The result is: 62.0
```

## 🎯 What Can the Chatbot Do?

### General Capabilities
- ✅ Answer questions in multiple languages
- ✅ Provide information on various topics
- ✅ Explain concepts and ideas
- ✅ Perform simple calculations
- ✅ Have natural conversations
- ✅ Provide recommendations
- ✅ Tell time and date

### Knowledge Domains
- Technology & Programming
- Science & Health
- Mathematics
- General Knowledge
- Current Affairs (with API integration)
- And more!

## 🛠️ Technical Details

### Architecture

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Language       │
│  Detection      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AI Backend     │
│  (Fallback/     │
│   Ollama/HF)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Response       │
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Display to     │
│  User           │
└─────────────────┘
```

### Files

- `app/ai_chatbot.py` - Streamlit web interface
- `app/chatbot_cli.py` - Command-line interface
- `CHATBOT_README.md` - This documentation

## 🎨 Customization

### Adding New Languages

Edit the `supported_languages` list in `ai_chatbot.py`:

```python
self.supported_languages = [
    "English", "Spanish", "French",
    # Add your language here
    "YourLanguage"
]
```

### Adding Custom Responses

Edit the `get_fallback_response` method to add custom logic:

```python
if "your_keyword" in user_message_lower:
    return "Your custom response"
```

### Changing AI Models

For Ollama, change the model in the request:

```python
payload = {
    "model": "mistral",  # Change to your preferred model
    "prompt": prompt,
    "stream": False
}
```

## 📊 Performance

- **Response Time**: 
  - Fallback: < 100ms
  - Ollama: 1-5 seconds (depends on model and hardware)
  - HuggingFace: 2-10 seconds (depends on API load)

- **Memory Usage**:
  - Web Interface: ~200MB
  - CLI: ~50MB
  - Ollama (with model loaded): ~4-8GB

## 🔒 Privacy & Security

- **Fallback Mode**: All processing is local, no data sent externally
- **Ollama**: Runs completely locally, no data leaves your machine
- **HuggingFace**: Data sent to HuggingFace servers (check their privacy policy)

## 🐛 Troubleshooting

### Issue: "Ollama is not running"
**Solution**: 
```bash
# Check if Ollama is installed
ollama --version

# Start Ollama service
ollama serve
```

### Issue: "HuggingFace API key not found"
**Solution**: 
```bash
# Set environment variable
$env:HF_API_KEY="your_key"

# Or add to .env file
echo "HF_API_KEY=your_key" > .env
```

### Issue: Streamlit not found
**Solution**:
```bash
pip install streamlit
```

## 🚀 Advanced Features

### Save Conversation History

The CLI version automatically saves conversations to JSON files:
```
chat_history_20251008_153000.json
```

### Custom System Prompts

Modify the system prompt in the code to change chatbot behavior:

```python
system_prompt = f"""You are a helpful assistant specialized in [YOUR DOMAIN].
Always respond in {language} language."""
```

## 📝 Examples by Language

### English
```
User: What is artificial intelligence?
Bot: Artificial Intelligence (AI) refers to computer systems that can perform tasks that typically require human intelligence...
```

### Spanish
```
User: ¿Qué es la inteligencia artificial?
Bot: La Inteligencia Artificial (IA) se refiere a sistemas informáticos que pueden realizar tareas que típicamente requieren inteligencia humana...
```

### Hindi
```
User: आर्टिफिशियल इंटेलिजेंस क्या है?
Bot: आर्टिफिशियल इंटेलिजेंस (AI) कंप्यूटर सिस्टम को संदर्भित करता है जो ऐसे कार्य कर सकते हैं जिनके लिए आमतौर पर मानव बुद्धि की आवश्यकता होती है...
```

## 🤝 Contributing

Feel free to extend the chatbot with:
- More languages
- Better responses
- Additional AI backends
- New features

## 📄 License

This project is part of the CMR Hackathon and follows the repository's license.

## 🙏 Acknowledgments

- Streamlit for the web framework
- Ollama for local LLM support
- HuggingFace for AI models
- The open-source community

---

**Built with ❤️ for the CMR Hackathon**
