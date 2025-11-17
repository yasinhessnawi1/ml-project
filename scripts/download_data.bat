@echo off
REM ==========================================
REM MM-IMDb Dataset Download Script (Windows)
REM ==========================================

echo.
echo ==========================================
echo MM-IMDb Dataset Download Script
echo ==========================================
echo.

REM Set variables
set DATA_DIR=data\raw
set DOWNLOAD_URL=https://www.kaggle.com/api/v1/datasets/download/johnarevalo/mmimdb
set ZIP_FILE=%DATA_DIR%\mmimdb.zip

REM Create data directory
echo Creating data directory: %DATA_DIR%
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Check for curl (Windows 10+ has curl built-in)
where curl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: curl not found. Please install curl or download the dataset manually.
    echo Download URL: %DOWNLOAD_URL%
    echo.
    echo Alternative: Install curl or use PowerShell to download:
    echo   powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%ZIP_FILE%'"
    pause
    exit /b 1
)

REM Download dataset
echo.
echo Downloading MM-IMDb dataset...
echo This may take several minutes depending on your connection...
echo.

curl -L -o "%ZIP_FILE%" "%DOWNLOAD_URL%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Download failed!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Download complete!
echo.

REM Extract dataset
echo Extracting dataset...
echo.

REM Check for tar (Windows 10+ has tar built-in for zip extraction)
where tar >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Use tar to extract (available on Windows 10+)
    tar -xf "%ZIP_FILE%" -C "%DATA_DIR%"
) else (
    REM Check for PowerShell (for Expand-Archive)
    where powershell >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%DATA_DIR%' -Force"
    ) else (
        echo.
        echo Warning: No extraction tool found (tar or PowerShell).
        echo Please manually extract: %ZIP_FILE%
        echo To: %DATA_DIR%
        pause
        exit /b 1
    )
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Extraction failed!
    pause
    exit /b 1
)

echo.
echo Extraction complete!
echo.

REM Clean up
echo Cleaning up...
if exist "%ZIP_FILE%" del /q "%ZIP_FILE%"

REM Verify extraction
echo.
echo Verifying extraction...
if exist "%DATA_DIR%\multimodal_imdb.hdf5" (
    echo Success! Found multimodal_imdb.hdf5
) else (
    echo Warning: multimodal_imdb.hdf5 not found in expected location.
)

if exist "%DATA_DIR%\metadata.npy" (
    echo Success! Found metadata.npy
) else (
    echo Warning: metadata.npy not found in expected location.
)

REM Show directory contents
echo.
echo Contents of %DATA_DIR%:
dir /b "%DATA_DIR%"

echo.
echo ==========================================
echo Download and extraction complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate your virtual environment:
echo    - venv: venv\Scripts\activate
echo    - conda: conda activate your_env_name
echo.
echo 2. Run data preprocessing:
echo    python scripts\preprocess_data.py
echo.
echo 3. Or run the test script:
echo    python test_implementation.py
echo.
echo ==========================================
echo.

pause
