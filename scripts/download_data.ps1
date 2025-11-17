# ==========================================
# MM-IMDb Dataset Download Script (PowerShell)
# ==========================================

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MM-IMDb Dataset Download Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Set variables
$DataDir = "data\raw"
$DownloadUrl = "https://www.kaggle.com/api/v1/datasets/download/johnarevalo/mmimdb"
$ZipFile = Join-Path $DataDir "mmimdb.zip"

# Create data directory
Write-Host "Creating data directory: $DataDir" -ForegroundColor Yellow
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}

# Download dataset
Write-Host ""
Write-Host "Downloading MM-IMDb dataset..." -ForegroundColor Yellow
Write-Host "This may take several minutes depending on your connection..." -ForegroundColor Gray
Write-Host ""

try {
    # Use .NET WebClient for progress bar
    $webClient = New-Object System.Net.WebClient

    # Register progress event
    Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -SourceIdentifier WebClient.DownloadProgressChanged -Action {
        $progressPercent = $EventArgs.ProgressPercentage
        Write-Progress -Activity "Downloading MM-IMDb dataset" -Status "$progressPercent% Complete" -PercentComplete $progressPercent
    } | Out-Null

    # Download file
    $webClient.DownloadFile($DownloadUrl, $ZipFile)

    # Unregister event
    Unregister-Event -SourceIdentifier WebClient.DownloadProgressChanged -ErrorAction SilentlyContinue

    Write-Host ""
    Write-Host "Download complete!" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "Error: Download failed!" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check your internet connection and try again." -ForegroundColor Yellow
    Write-Host "Or download manually from: $DownloadUrl" -ForegroundColor Yellow
    exit 1
}

# Extract dataset
Write-Host "Extracting dataset..." -ForegroundColor Yellow
Write-Host ""

try {
    # Use PowerShell's Expand-Archive
    Expand-Archive -Path $ZipFile -DestinationPath $DataDir -Force

    Write-Host ""
    Write-Host "Extraction complete!" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "Error: Extraction failed!" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please manually extract: $ZipFile" -ForegroundColor Yellow
    Write-Host "To: $DataDir" -ForegroundColor Yellow
    exit 1
}

# Clean up
Write-Host "Cleaning up..." -ForegroundColor Yellow
if (Test-Path $ZipFile) {
    Remove-Item $ZipFile -Force
}

# Verify extraction
Write-Host ""
Write-Host "Verifying extraction..." -ForegroundColor Yellow

$hdf5File = Join-Path $DataDir "multimodal_imdb.hdf5"
$metadataFile = Join-Path $DataDir "metadata.npy"

if (Test-Path $hdf5File) {
    $fileSize = (Get-Item $hdf5File).Length / 1GB
    Write-Host "Success! Found multimodal_imdb.hdf5 ($($fileSize.ToString('F2')) GB)" -ForegroundColor Green
} else {
    Write-Host "Warning: multimodal_imdb.hdf5 not found in expected location." -ForegroundColor Yellow
}

if (Test-Path $metadataFile) {
    $fileSize = (Get-Item $metadataFile).Length / 1MB
    Write-Host "Success! Found metadata.npy ($($fileSize.ToString('F2')) MB)" -ForegroundColor Green
} else {
    Write-Host "Warning: metadata.npy not found in expected location." -ForegroundColor Yellow
}

# Show directory contents
Write-Host ""
Write-Host "Contents of $DataDir`:" -ForegroundColor Cyan
Get-ChildItem $DataDir | Format-Table Name, Length, LastWriteTime -AutoSize

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Download and extraction complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate your virtual environment:" -ForegroundColor White
Write-Host "   - venv: .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "   - conda: conda activate your_env_name" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run data preprocessing:" -ForegroundColor White
Write-Host "   python scripts\preprocess_data.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or run the test script:" -ForegroundColor White
Write-Host "   python test_implementation.py" -ForegroundColor Gray
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
