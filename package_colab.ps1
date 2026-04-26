# package_colab.ps1
# Run this script to generate a clean ZIP file of the backend for Colab uploading.

Write-Host "Packaging backend for Google Colab..."
$sourcePath = "$PSScriptRoot\backend\*"
$destinationPath = "$PSScriptRoot\backend_colab.zip"

If (Test-Path $destinationPath) {
    Remove-Item $destinationPath -Force
}

# The wildcard (\*) ensures ONLY the contents of the folder are zipped, preventing the double-folder issue.
Compress-Archive -Path $sourcePath -DestinationPath $destinationPath -Force

Write-Host "--------------------------------------------------------"
Write-Host "Success! backend_colab.zip created at $destinationPath"
Write-Host "Upload this to Colab and extract it by running:"
Write-Host "   !unzip -q -o /content/backend_colab.zip -d /content/backend"
Write-Host "--------------------------------------------------------"
