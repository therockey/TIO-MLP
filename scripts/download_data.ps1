# Pobiera oryginalne pliki MNIST (format IDX) do data\MNIST\raw.
# Uzycie:
#   powershell -ExecutionPolicy Bypass -File .\scripts\download_data.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\download_data.ps1 -DataDir C:\mnist
[CmdletBinding()]
param(
    [string]$DataDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'data')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    # nowsze wersje PowerShell nie wymagaja tego ustawienia
}

$rawDir = Join-Path (Join-Path $DataDir 'MNIST') 'raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$mirrors = @(
    'https://ossci-datasets.s3.amazonaws.com/mnist',
    'https://storage.googleapis.com/cvdf-datasets/mnist',
    'http://yann.lecun.com/exdb/mnist'
)

$files = [ordered]@{
    'train-images-idx3-ubyte.gz' = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
    'train-labels-idx1-ubyte.gz' = 'd53e105ee54ea40749a09fcbcd1e9432'
    't10k-images-idx3-ubyte.gz'  = '9fb629c4189551a2d022fa330f9573f3'
    't10k-labels-idx1-ubyte.gz'  = 'ec29112dd5afa0611ce80d1b7f02629c'
}

function Expand-Gzip {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    try {
        Add-Type -AssemblyName System.IO.Compression -ErrorAction Stop
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
    } catch {
        # w PowerShell 7 typy sa dostepne bez ladowania assembly
    }

    $inputStream = [System.IO.File]::OpenRead($Source)
    $outputStream = [System.IO.File]::Create($Destination)
    try {
        $gzipStream = New-Object System.IO.Compression.GZipStream -ArgumentList @(
            $inputStream,
            [System.IO.Compression.CompressionMode]::Decompress
        )
        try {
            $gzipStream.CopyTo($outputStream)
        } finally {
            $gzipStream.Dispose()
        }
    } finally {
        $outputStream.Dispose()
        $inputStream.Dispose()
    }
}

Write-Host "Katalog danych: $rawDir"

foreach ($file in $files.Keys) {
    $gz = Join-Path $rawDir $file
    $out = $gz.Substring(0, $gz.Length - 3)
    $expected = $files[$file]
    $needDownload = $true

    if (Test-Path -LiteralPath $gz) {
        $actual = (Get-FileHash -LiteralPath $gz -Algorithm MD5).Hash.ToLower()
        if ($actual -eq $expected) {
            Write-Host "[OK] $file"
            $needDownload = $false
        }
    }

    if ($needDownload) {
        Write-Host "[POBIERANIE] $file"
        if (Test-Path -LiteralPath $gz) {
            Remove-Item -LiteralPath $gz -Force
        }

        $downloaded = $false
        foreach ($mirror in $mirrors) {
            $url = "$mirror/$file"
            Write-Host "  probuje: $url"
            try {
                Invoke-WebRequest -Uri $url -OutFile $gz -UseBasicParsing
                $actual = (Get-FileHash -LiteralPath $gz -Algorithm MD5).Hash.ToLower()
                if ($actual -ne $expected) {
                    throw "Zly MD5: oczekiwano $expected, jest $actual"
                }
                $downloaded = $true
                break
            } catch {
                Write-Warning "  nie udalo sie: $($_.Exception.Message)"
                if (Test-Path -LiteralPath $gz) {
                    Remove-Item -LiteralPath $gz -Force
                }
            }
        }

        if (-not $downloaded) {
            throw "Nie udalo sie pobrac $file z zadnego mirrora."
        }
        Write-Host "[OK] $file (MD5)"
    }

    if (-not (Test-Path -LiteralPath $out)) {
        Write-Host "[ROZPAKOWANIE] $file"
        try {
            Expand-Gzip -Source $gz -Destination $out
        } catch {
            if (Test-Path -LiteralPath $out) {
                Remove-Item -LiteralPath $out -Force
            }
            throw
        }
    }
}

Write-Host ""
Write-Host "Gotowe. Pliki MNIST sa w: $rawDir"
Write-Host "Mozesz uruchomic: python run.py train --epochs 10"
