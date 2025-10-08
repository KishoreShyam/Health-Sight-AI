@echo off
echo ========================================
echo   AI Multilingual Chatbot Launcher
echo ========================================
echo.
echo Select an option:
echo   1. Web Interface (Streamlit)
echo   2. Command-Line Interface
echo   3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting web interface...
    echo Open your browser to: http://localhost:8501
    echo.
    streamlit run app/ai_chatbot.py
) else if "%choice%"=="2" (
    echo.
    echo Starting command-line chatbot...
    echo.
    python app/chatbot_cli.py
) else if "%choice%"=="3" (
    echo.
    echo Goodbye!
    exit
) else (
    echo.
    echo Invalid choice. Please run again.
    pause
)
