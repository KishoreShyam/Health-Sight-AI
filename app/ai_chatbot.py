"""
AI Multilingual Chatbot
Supports multiple languages and can answer various questions
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict
import os

# Configure page
st.set_page_config(
    page_title="AI Multilingual Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 5px solid #8bc34a;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #666;
    }
    .message-content {
        font-size: 1rem;
        line-height: 1.6;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #999;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


class MultilingualChatbot:
    """AI Chatbot with multilingual support"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.supported_languages = [
            "English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Russian", "Chinese", "Japanese", "Korean",
            "Arabic", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi"
        ]
    
    def get_response_ollama(self, user_message: str, language: str = "English") -> str:
        """
        Get response from Ollama (local LLM)
        Requires Ollama to be installed and running locally
        """
        try:
            url = "http://localhost:11434/api/generate"
            
            system_prompt = f"""You are a helpful, friendly, and knowledgeable AI assistant. 
You can answer questions on any topic. Always respond in {language} language.
Be concise, accurate, and helpful. If you don't know something, admit it honestly."""
            
            prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            payload = {
                "model": "llama2",  # or "mistral", "codellama", etc.
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "I couldn't generate a response.")
            else:
                return f"Error: Unable to connect to Ollama (Status: {response.status_code})"
                
        except requests.exceptions.ConnectionError:
            return "⚠️ Ollama is not running. Please start Ollama or use the fallback mode."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_response_huggingface(self, user_message: str, language: str = "English") -> str:
        """
        Get response from HuggingFace Inference API
        Requires HF_API_KEY environment variable
        """
        try:
            api_key = os.getenv("HF_API_KEY")
            if not api_key:
                return "⚠️ HuggingFace API key not found. Set HF_API_KEY environment variable."
            
            url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            system_prompt = f"You are a helpful AI assistant. Respond in {language}."
            prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated.")
                return str(result)
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_fallback_response(self, user_message: str, language: str = "English") -> str:
        """
        Fallback response system with predefined answers
        """
        user_message_lower = user_message.lower()
        
        # Greeting responses
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greeting in user_message_lower for greeting in greetings):
            responses = {
                "English": "Hello! I'm your AI assistant. How can I help you today?",
                "Spanish": "¡Hola! Soy tu asistente de IA. ¿Cómo puedo ayudarte hoy?",
                "French": "Bonjour! Je suis votre assistant IA. Comment puis-je vous aider aujourd'hui?",
                "German": "Hallo! Ich bin Ihr KI-Assistent. Wie kann ich Ihnen heute helfen?",
                "Hindi": "नमस्ते! मैं आपका AI सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
                "Chinese": "你好！我是你的AI助手。今天我能帮你什么？",
                "Japanese": "こんにちは！私はあなたのAIアシスタントです。今日はどのようにお手伝いできますか？"
            }
            return responses.get(language, responses["English"])
        
        # Help/capability questions
        help_keywords = ["what can you do", "help", "capabilities", "features"]
        if any(keyword in user_message_lower for keyword in help_keywords):
            return f"""I'm a multilingual AI assistant! I can help you with:

✅ Answering general knowledge questions
✅ Explaining concepts and topics
✅ Providing information on various subjects
✅ Having conversations in {language}
✅ Assisting with problem-solving
✅ Offering recommendations and suggestions

Just ask me anything, and I'll do my best to help!"""
        
        # Name questions
        if "your name" in user_message_lower or "who are you" in user_message_lower:
            return f"I'm an AI chatbot assistant designed to help answer your questions in multiple languages, including {language}!"
        
        # Time/date questions
        if "time" in user_message_lower or "date" in user_message_lower:
            now = datetime.now()
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Default response
        return f"""I'm currently running in fallback mode. To get better responses, please:

1. **Install Ollama**: Download from https://ollama.ai and run `ollama pull llama2`
2. **Use HuggingFace API**: Set your HF_API_KEY environment variable

For now, I can answer basic questions. What would you like to know?"""
    
    def get_response(self, user_message: str, language: str = "English", mode: str = "fallback") -> str:
        """
        Main method to get chatbot response
        """
        if mode == "ollama":
            return self.get_response_ollama(user_message, language)
        elif mode == "huggingface":
            return self.get_response_huggingface(user_message, language)
        else:
            return self.get_fallback_response(user_message, language)
    
    def add_to_history(self, role: str, message: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })


def display_message(role: str, message: str, timestamp: str):
    """Display a chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{message}</div>
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-header">🤖 AI Assistant</div>
            <div class="message-content">{message}</div>
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MultilingualChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Language selection
        language = st.selectbox(
            "🌍 Response Language",
            st.session_state.chatbot.supported_languages,
            index=0
        )
        
        # Mode selection
        st.subheader("🔧 AI Mode")
        mode = st.radio(
            "Select AI backend:",
            ["fallback", "ollama", "huggingface"],
            format_func=lambda x: {
                "fallback": "💡 Fallback (No setup required)",
                "ollama": "🦙 Ollama (Local)",
                "huggingface": "🤗 HuggingFace API"
            }[x]
        )
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📖 Setup Instructions")
        
        if mode == "ollama":
            st.info("""
            **Ollama Setup:**
            1. Install from https://ollama.ai
            2. Run: `ollama pull llama2`
            3. Ensure Ollama is running
            """)
        elif mode == "huggingface":
            st.info("""
            **HuggingFace Setup:**
            1. Get API key from huggingface.co
            2. Set environment variable:
               `HF_API_KEY=your_key`
            """)
        else:
            st.success("✅ Fallback mode requires no setup!")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chatbot.conversation_history = []
            st.rerun()
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Language", language)
    
    # Main chat interface
    st.title("🤖 AI Multilingual Chatbot")
    st.markdown(f"**Current Language:** {language} | **Mode:** {mode.upper()}")
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            display_message(msg["role"], msg["message"], msg["timestamp"])
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            key="user_input",
            placeholder=f"Type your message in any language...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "message": user_input,
            "timestamp": timestamp
        })
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            bot_response = st.session_state.chatbot.get_response(
                user_input,
                language=language,
                mode=mode
            )
        
        # Add bot message
        bot_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "bot",
            "message": bot_response,
            "timestamp": bot_timestamp
        })
        
        # Rerun to update chat
        st.rerun()
    
    # Welcome message if no messages
    if len(st.session_state.messages) == 0:
        st.info(f"""
        👋 **Welcome to the AI Multilingual Chatbot!**
        
        I can help you with:
        - Answering questions in **{language}**
        - General knowledge and information
        - Explanations and discussions
        - Problem-solving assistance
        
        Just type your message below and press Send!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with ❤️ using Streamlit | Supports 15+ languages
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
