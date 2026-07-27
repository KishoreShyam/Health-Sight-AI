@echo off
echo ===================================================
echo   Health Sight AI - Azure Web App Deployment Script
echo ===================================================
echo.

echo Checking Azure CLI installation...
where az >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Azure CLI is not installed. 
    echo Please install Azure CLI from: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
    pause
    exit /b 1
)

echo 1. Logging into Azure...
call az login

echo 2. Creating Resource Group (healthsight-rg in centralindia)...
call az group create --name healthsight-rg --location centralindia

echo 3. Creating App Service Plan (B1 Linux)...
call az appservice plan create --name healthsight-plan --resource-group healthsight-rg --sku B1 --is-linux

echo 4. Provisioning Azure Web App (healthsightai)...
call az webapp create --name healthsightai --resource-group healthsight-rg --plan healthsight-plan --runtime "PYTHON:3.11" --startup-file "startup.sh"

echo 5. Injecting Environment Variables into Azure App Settings...
call az webapp config appsettings set --resource-group healthsight-rg --name healthsightai --settings AZURE_ENDPOINT="YOUR_AZURE_ENDPOINT" AZURE_KEY="YOUR_AZURE_KEY" AZURE_STORAGE_CONNECTION_STRING="YOUR_STORAGE_CONNECTION_STRING" AZURE_STORAGE_CONTAINER="health-sight-uploads" AZURE_SPEECH_REGION="centralindia" SCM_DO_BUILD_DURING_DEPLOYMENT="true"

echo 6. Deploying Application to Azure...
call az webapp up --name healthsightai --resource-group healthsight-rg --plan healthsight-plan

echo.
echo ===================================================
echo   Deployment Complete!
echo   Your live application URL:
echo   https://healthsightai.azurewebsites.net
echo ===================================================
pause
