"""
Command-Line AI Chatbot
Simple terminal-based multilingual chatbot
"""

import sys
import os
from datetime import datetime
import requests
import json


class SimpleChatbot:
    """Simple command-line chatbot"""
    
    def __init__(self):
        self.conversation_history = []
        self.languages = {
            "1": "English",
            "2": "Spanish", 
            "3": "French",
            "4": "German",
            "5": "Hindi",
            "6": "Chinese",
            "7": "Japanese",
            "8": "Arabic"
        }
    
    def get_response(self, user_message: str, language: str = "English") -> str:
        """Generate response based on user input"""
        user_message_lower = user_message.lower()
        
        # Greetings
        greetings = ["hello", "hi", "hey", "greetings", "namaste", "hola", "bonjour"]
        if any(greeting in user_message_lower for greeting in greetings):
            responses = {
                "English": "Hello! How can I help you today?",
                "Spanish": "¡Hola! ¿Cómo puedo ayudarte hoy?",
                "French": "Bonjour! Comment puis-je vous aider aujourd'hui?",
                "German": "Hallo! Wie kann ich Ihnen heute helfen?",
                "Hindi": "नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूं?",
                "Chinese": "你好！今天我能帮你什么？",
                "Japanese": "こんにちは！今日はどのようにお手伝いできますか？",
                "Arabic": "مرحبا! كيف يمكنني مساعدتك اليوم؟"
            }
            return responses.get(language, responses["English"])
        
        # Help
        if "help" in user_message_lower or "what can you do" in user_message_lower:
            return f"""I'm an AI assistant that can help you with:
• Answering questions in {language}
• General knowledge and information
• Explaining concepts
• Having conversations
• Providing recommendations

Just ask me anything!"""
        
        # Time/Date
        if "time" in user_message_lower or "date" in user_message_lower:
            now = datetime.now()
            return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Math operations
        if any(op in user_message_lower for op in ["calculate", "what is", "solve"]):
            try:
                # Extract numbers and operators
                import re
                numbers = re.findall(r'\d+\.?\d*', user_message)
                if len(numbers) >= 2:
                    if "+" in user_message or "plus" in user_message or "add" in user_message:
                        result = float(numbers[0]) + float(numbers[1])
                        return f"The result is: {result}"
                    elif "-" in user_message or "minus" in user_message or "subtract" in user_message:
                        result = float(numbers[0]) - float(numbers[1])
                        return f"The result is: {result}"
                    elif "*" in user_message or "×" in user_message or "multiply" in user_message:
                        result = float(numbers[0]) * float(numbers[1])
                        return f"The result is: {result}"
                    elif "/" in user_message or "÷" in user_message or "divide" in user_message:
                        if float(numbers[1]) != 0:
                            result = float(numbers[0]) / float(numbers[1])
                            return f"The result is: {result}"
                        else:
                            return "Cannot divide by zero!"
            except:
                pass
        
        # Weather (mock)
        if "weather" in user_message_lower:
            return "I don't have real-time weather data, but you can check weather.com or your local weather service!"
        
        # About
        if "who are you" in user_message_lower or "your name" in user_message_lower:
            return f"I'm an AI chatbot assistant. I can communicate in multiple languages including {language}!"
        
        # Goodbye
        if any(word in user_message_lower for word in ["bye", "goodbye", "exit", "quit"]):
            return "Goodbye! Have a great day! 👋"
        
        # Knowledge base responses
        knowledge = {
            "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
            "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
            "machine learning": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
            "health": "Health is a state of complete physical, mental, and social well-being. Regular exercise, balanced diet, and adequate sleep are key to maintaining good health.",
            "cancer": "Cancer is a disease where abnormal cells divide uncontrollably. Early detection through screening and advances in treatment have improved survival rates significantly.",
        }
        
        for keyword, response in knowledge.items():
            if keyword in user_message_lower:
                return response
        
        # Default response
        return f"""I understand you're asking about: "{user_message}"

While I'm running in basic mode, I can help with:
- General questions
- Simple calculations
- Information about common topics
- Conversations in {language}

Could you rephrase your question or ask something else?"""
    
    def chat(self):
        """Main chat loop"""
        print("\n" + "="*60)
        print("🤖 AI MULTILINGUAL CHATBOT")
        print("="*60)
        print("\nSelect your preferred language:")
        for key, lang in self.languages.items():
            print(f"  {key}. {lang}")
        
        lang_choice = input("\nEnter number (default: 1 for English): ").strip() or "1"
        selected_language = self.languages.get(lang_choice, "English")
        
        print(f"\n✅ Language set to: {selected_language}")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("Type 'help' to see what I can do.")
        print("-"*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n🤖 Bot: Goodbye! Have a great day! 👋\n")
                    break
                
                # Get response
                response = self.get_response(user_input, selected_language)
                
                # Display response
                print(f"\n🤖 Bot: {response}\n")
                
                # Add to history
                self.conversation_history.append({
                    "user": user_input,
                    "bot": response,
                    "timestamp": datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Goodbye! 👋\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
        
        # Save conversation history
        if self.conversation_history:
            try:
                filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
                print(f"💾 Conversation saved to: {filename}\n")
            except:
                pass


def main():
    """Run the chatbot"""
    chatbot = SimpleChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
