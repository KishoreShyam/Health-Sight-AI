@echo off
echo ================================================================================
echo HEALTH SIGHT AI - PUSH TO GITHUB
echo ================================================================================
echo.
echo Repository: https://github.com/KishoreShyam/Health-Sight-AI.git
echo.
echo This will push your project to GitHub (excluding large files)
echo.
echo ================================================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [1/6] Checking Git configuration...
git config user.name >nul 2>&1
if errorlevel 1 (
    echo.
    echo Git user not configured. Please enter your details:
    set /p username="Enter your GitHub username: "
    set /p email="Enter your GitHub email: "
    git config --global user.name "%username%"
    git config --global user.email "%email%"
    echo Configuration saved!
)

echo [2/6] Initializing Git repository...
if not exist .git (
    git init
    echo Repository initialized
) else (
    echo Repository already initialized
)

echo [3/6] Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/KishoreShyam/Health-Sight-AI.git
echo Remote added: https://github.com/KishoreShyam/Health-Sight-AI.git

echo [4/6] Staging files...
git add .
echo Files staged (large files excluded by .gitignore)

echo [5/6] Committing changes...
git commit -m "Initial commit: Health Sight AI - Multimodal Cancer Detection System"
if errorlevel 1 (
    echo No changes to commit or commit failed
)

echo [6/6] Pushing to GitHub...
echo.
echo NOTE: You may need to enter your GitHub credentials
echo       Use Personal Access Token (PAT) instead of password
echo.
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo PUSH FAILED
    echo ================================================================================
    echo.
    echo Possible reasons:
    echo   1. Authentication failed - Use Personal Access Token
    echo   2. Repository already has content - Try: git pull origin main --allow-unrelated-histories
    echo   3. Network issues - Check your internet connection
    echo.
    echo Manual commands:
    echo   git pull origin main --allow-unrelated-histories
    echo   git push -u origin main --force
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Project pushed to GitHub
echo ================================================================================
echo.
echo View your repository at:
echo https://github.com/KishoreShyam/Health-Sight-AI
echo.
echo Next steps:
echo   1. Go to GitHub and verify all files are there
echo   2. Add repository description and topics
echo   3. Update README.md if needed
echo.
pause
