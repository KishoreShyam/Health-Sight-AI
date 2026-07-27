import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

# Neural Voice mappings for different languages
VOICES = {
    "English": "en-US-JennyNeural",
    "Tamil": "ta-IN-PallaviNeural",
    "Telugu": "te-IN-ShrutiNeural",
    "Hindi": "hi-IN-SwararaNeural",
    "Kannada": "kn-IN-SapnaNeural",
    "Malayalam": "ml-IN-SobhanaNeural"
}

# High-quality medically accurate pre-translated diagnostic warnings and explanations
# This ensures perfect grammar and professional terminology without API translation errors.
TRANSLATIONS = {
    "English": {
        "benign": "The AI analysis predicts a low risk of skin cancer. The lesion appears benign. However, please monitor for any changes and consult a doctor if you notice any evolving symptoms.",
        "malignant": "The AI analysis predicts a high risk of skin cancer. Please consult a dermatologist or healthcare professional immediately for a detailed clinical examination."
    },
    "Tamil": {
        "benign": "செயற்கை நுண்ணறிவு பகுப்பாய்வு தோல் புற்றுநோய்க்கான குறைந்த அபாயத்தை கணிக்கிறது. இந்த தழும்பு தீங்கற்றதாக தோன்றுகிறது. இருப்பினும், ஏதேனும் மாற்றங்களை கண்காணித்து, அறிகுறிகள் தென்பட்டால் மருத்துவரை அணுகவும்.",
        "malignant": "செயற்கை நுண்ணறிவு பகுப்பாய்வு தோல் புற்றுநோய்க்கான அதிக அபாயத்தை கணிக்கிறது. விரிவான மருத்துவ பரிσοதனைக்கு உடனடியாக ஒரு தோல் மருத்துவர் அல்லது சுகாதார நிபுணரை அணுகவும்."
    },
    "Hindi": {
        "benign": "एआई विश्लेषण त्वचा कैंसर के कम जोखिम की भविष्यवाणी करता है। यह घाव सौम्य प्रतीत होता है। हालांकि, किसी भी बदलाव पर नजर रखें और लक्षण दिखने पर डॉक्टर से सलाह लें।",
        "malignant": "एआई विश्लेषण त्वचा कैंसर के उच्च जोखिम की भविष्यवाणी करता है। विस्तृत नैदानिक परीक्षण के लिए कृपया तुरंत त्वचा विशेषज्ञ या स्वास्थ्य पेशेवर से परामर्श लें।"
    },
    "Telugu": {
        "benign": "ఏఐ విశ్లేషణ చర్మ క్యాన్సర్ యొక్క తక్కువ ప్రమాదాన్ని అంచనా వేస్తుంది. ఈ గాయం ప్రమాదకరం కానిదిగా కనిపిస్తుంది. అయినప్పటికీ, ఏవైనా మార్పులను గమనిస్తూ ఉండండి మరియు వైద్యుడిని సంప్రదించండి.",
        "malignant": "ఏఐ విశ్లేషణ చర్మ క్యాన్సర్ యొక్క అధిక ప్రమాదాన్ని అంచనా వేస్తుంది. వివరణాत्मक వైద్య పరీక్ష కోసం దయచేసి వెంటనే చర్మవ్యాధి నిపుణుడిని లేదా ఆరోగ్య నిపుణుడిని సంప్రదించండి।"
    },
    "Kannada": {
        "benign": "ಎಐ ವಿಶ್ಲೇಷಣೆಯು ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್‌ನ ಕಡಿಮೆ ಅಪಾಯವನ್ನು ಸೂಚಿಸುತ್ತದೆ. ಈ ಗಾಯವು ಅಪಾಯಕಾರಿಯಲ್ಲದಂತೆ ಕಾಣುತ್ತದೆ. ಆದಾಗ್ಯೂ, ಯಾವುದೇ ಬದಲಾವಣೆಗಳನ್ನು ಗಮನಿಸಿ ಮತ್ತು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "malignant": "ಎಐ ವಿಶ್ಲೇಷಣೆಯು ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್‌ನ ಹೆಚ್ಚಿನ ಅಪಾಯವನ್ನು ಸೂಚಿಸುತ್ತದೆ. ವಿವರವಾದ ವೈದ್ಯಕೀಯ ತಪಾಸಣೆಗಾಗಿ ದಯವಿಟ್ಟು ತಕ್ಷಣವೇ ಚರ್ಮರೋಗ ವೈದ್ಯರನ್ನು ಅಥವಾ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ।"
    },
    "Malayalam": {
        "benign": "എഐ വിശകലനം ചർമ്മ അർബുദത്തിനുള്ള കുറഞ്ഞ സാധ്യത പ്രവചിക്കുന്നു. ഈ വ്രണം ദോഷകരമല്ലെന്ന് തോന്നുന്നു. എങ്കിലും, മാറ്റങ്ങൾ നിരീക്ഷിക്കുകയും ലക്ഷണങ്ങൾ കണ്ടാൽ ഡോക്ടറെ സമീപിക്കുകയും ചെയ്യുക.",
        "malignant": "എഐ വിശകലനം ചർമ്മ അർബുദത്തിനുള്ള ഉയർന്ന സാധ്യത പ്രവചിക്കുന്നു. വിശദമായ പരിശോധനയ്ക്കായി ദയവായി ഉടൻ തന്നെ ഒരു ചർമ്മരോഗ വിദഗ്ദ്ധനെയോ ആരോഗ്യ പ്രവർത്തകനെയോ സമീപിക്കുക."
    }
}

def get_voice_script(prediction_label, language="English"):
    """
    Retrieves the correct medical explanation based on prediction status and selected language.
    """
    lang_data = TRANSLATIONS.get(language, TRANSLATIONS["English"])
    label_key = "benign" if "benign" in prediction_label.lower() else "malignant"
    return lang_data.get(label_key, lang_data["benign"])

def synthesize_speech(text, language="English"):
    """
    Synthesizes the given text to speech using Azure Speech Services.
    Returns the raw audio bytes (WAV format).
    """
    key = os.getenv("AZURE_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "centralindia")
    
    if not key:
        raise ValueError("Azure API Key is not configured. Please check your .env file.")
        
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    
    # Resolve and set the neural voice name
    voice_name = VOICES.get(language, VOICES["English"])
    speech_config.speech_synthesis_voice_name = voice_name
    
    # Synthesize to an in-memory stream rather than directly to a speaker/file
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        error_msg = f"Speech synthesis canceled. Reason: {cancellation_details.reason}"
        if cancellation_details.error_details:
            error_msg += f" Details: {cancellation_details.error_details}"
        raise RuntimeError(error_msg)
    else:
        raise RuntimeError(f"Speech synthesis failed with reason: {result.reason}")
