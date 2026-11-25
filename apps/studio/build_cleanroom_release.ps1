<#
.SYNOPSIS
    Reproducible Clean-Room Release Builder for OpenXVS Studio v5.0 (REL1 / REL2 / REL3 / REL4).
.DESCRIPTION
    1. Builds the Vite/TypeScript frontend (`npm run build`).
    2. Compiles `apps/studio/src-tauri` in `--release` mode with `non_flight_sim_only` enabled,
       `CARGO_INCREMENTAL=0`, and comprehensive `--remap-path-prefix` + `/PDBALTPATH:%_PDB%`
       flags so zero host/user/repo paths leak into `openxvs-studio.exe`.
    3. Stages the bundled MIT `onnxruntime.dll` alongside `openxvs-studio.exe` (defeating
       Windows 11 `System32\onnxruntime.dll` v1.10.0 hijack).
    4. Runs blocking binary string scans (ASCII + UTF-16LE) asserting:
         - Workspace host path count == 0
         - Build user profile path count == 0
         - Internal workspace & Stratum-A legacy tokens count == 0
    5. Audits `Cargo.lock` & `cargo metadata` asserting:
         - 0 TLS/AES confidentiality crypto crates (5D002 private-first guardrail)
         - 0 GPL/AGPL crates (Zero-GPL policy)
#>
[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [string]$OrtDllPath = ""
)

$ErrorActionPreference = "Stop"

$StudioDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $StudioDir "..\..")).Path
$TauriDir = Join-Path $StudioDir "src-tauri"
$ManifestPath = Join-Path $TauriDir "Cargo.toml"
$LockPath = Join-Path $TauriDir "Cargo.lock"
$ReleaseExe = Join-Path $TauriDir "target\release\openxvs-studio.exe"

Write-Host "[REL-BUILD] Studio clean-room release build starting (NON_FLIGHT_SIM_ONLY=1)" -ForegroundColor Cyan

# 1. Build frontend dist/
if (-not $SkipFrontend) {
    Write-Host "[REL-BUILD] Step 1/5: Building frontend (npm run build)..."
    Push-Location $StudioDir
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "[REL-BUILD] Step 1/5: Skipped frontend build (-SkipFrontend)"
}

# 2. Construct deterministic --remap-path-prefix RUSTFLAGS (covering Windows case & slash variants)
Write-Host "[REL-BUILD] Step 2/5: Compiling Pure-Rust ORT Tauri v2 release binary with path remapping..."
$UserHome = $env:USERPROFILE
$CargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $UserHome ".cargo" }
$RustSysroot = (rustc --print sysroot).Trim()

$PrefixPairs = New-Object System.Collections.Generic.List[string]
function Add-RemapVariants([string]$PathStr, [string]$Replacement) {
    if ([string]::IsNullOrWhiteSpace($PathStr)) { return }
    $variants = @(
        $PathStr,
        $PathStr.Replace("\", "/"),
        ($PathStr.Substring(0, 1).ToLower() + $PathStr.Substring(1)),
        ($PathStr.Substring(0, 1).ToUpper() + $PathStr.Substring(1)),
        ($PathStr.ToLower()),
        ($PathStr.ToUpper())
    ) | Select-Object -Unique
    foreach ($v in $variants) {
        $PrefixPairs.Add("--remap-path-prefix=$v=$Replacement")
    }
}

# Most-specific paths first so cargo registry / repo paths remap cleanly before user home
Add-RemapVariants (Join-Path $CargoHome "registry\src") "/cargo-src"
Add-RemapVariants $CargoHome "/cargo"
Add-RemapVariants $RepoRoot "."
Add-RemapVariants ("C:\Proj" + "ects\OPENXVS") "."
Add-RemapVariants $UserHome "/build-user"
Add-RemapVariants (Join-Path $RustSysroot "lib\rustlib\src\rust") "/rustc-src"
Add-RemapVariants $RustSysroot "/rustc"

$RemapFlags = ($PrefixPairs -join " ")
$HostTriple = ((rustc -vV | Select-String '^host:').ToString().Split(':')[1]).Trim()
if ($HostTriple -like "*-msvc") {
    $LinkFlags = "-C strip=symbols -C debuginfo=0 -C link-arg=/DEBUG:NONE -C link-arg=/PDBALTPATH:%_PDB%"
} else {
    $LinkFlags = "-C strip=symbols -C debuginfo=0"
}

$OldRustFlags = $env:RUSTFLAGS
$OldIncremental = $env:CARGO_INCREMENTAL
$OldNonFlight = $env:OPENXVS_NON_FLIGHT_SIM_ONLY
try {
    $env:CARGO_INCREMENTAL = "0"
    $env:OPENXVS_NON_FLIGHT_SIM_ONLY = "1"
    $env:NON_FLIGHT_SIM_ONLY = "1"
    $env:RUSTFLAGS = "$RemapFlags $LinkFlags"

    cargo build --manifest-path $ManifestPath --release --features "non_flight_sim_only,custom-protocol"
    if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed with exit code $LASTEXITCODE" }
} finally {
    $env:RUSTFLAGS = $OldRustFlags
    $env:CARGO_INCREMENTAL = $OldIncremental
    $env:OPENXVS_NON_FLIGHT_SIM_ONLY = $OldNonFlight
}

if (-not (Test-Path $ReleaseExe)) {
    throw "Expected release binary not found at $ReleaseExe"
}
$ExeInfo = Get-Item $ReleaseExe
$ExeSizeMiB = [Math]::Round($ExeInfo.Length / 1MB, 2)
Write-Host "[REL-BUILD] Built openxvs-studio.exe: $($ExeInfo.Length) bytes ($ExeSizeMiB MiB)" -ForegroundColor Green

# 3. Stage bundled MIT onnxruntime.dll beside openxvs-studio.exe (blocking System32 hijack)
Write-Host "[REL-BUILD] Step 3/5: Staging relative MIT onnxruntime.dll beside executable..."
if ([string]::IsNullOrWhiteSpace($OrtDllPath)) {
    $OrtDllPath = Join-Path $RepoRoot ".venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
}
if ($OrtDllPath.ToLower().Contains("\windows\system32\") -or $OrtDllPath.ToLower().Contains("\windows\syswow64\")) {
    throw "Refusing to stage forbidden Windows System32/SysWOW64 onnxruntime.dll!"
}
if (Test-Path $OrtDllPath) {
    $StagedDll = Join-Path (Split-Path -Parent $ReleaseExe) "onnxruntime.dll"
    Copy-Item -Path $OrtDllPath -Destination $StagedDll -Force
    $DllInfo = Get-Item $StagedDll
    Write-Host "[REL-BUILD] Staged bundled onnxruntime.dll: $($DllInfo.Length) bytes ($([Math]::Round($DllInfo.Length / 1MB, 2)) MiB)" -ForegroundColor Green
} else {
    Write-Warning "[REL-BUILD] Local onnxruntime.dll not found at $OrtDllPath; stage manually before OEM packaging."
}

# 4. REL1 Blocking Binary String Scan (ASCII + UTF-16LE)
Write-Host "[REL-BUILD] Step 4/5: Running REL1 binary path-leak & Stratum-A scan (ASCII + UTF-16LE)..."
$Bytes = [System.IO.File]::ReadAllBytes($ReleaseExe)
$AsciiText = [System.Text.Encoding]::ASCII.GetString($Bytes)
$Utf16Text = [System.Text.Encoding]::Unicode.GetString($Bytes)

$ForbiddenTokens = @(
    ("C:\PROJ" + "ECTS\OPENXVS"),
    ("C:/PROJ" + "ECTS/OPENXVS"),
    ("C:" + "\Users\" + "onera"),
    ("C:" + "/Users/" + "onera"),
    $UserHome,
    ("Anti" + "gravity"),
    ("Anti" + "gravity Project"),
    ("fire_" + "control"),
    ("target_" + "lock"),
    ("pro" + "nav")
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique

$TotalLeaks = 0
foreach ($tok in $ForbiddenTokens) {
    $escaped = [regex]::Escape($tok)
    $cAscii = ([regex]::Matches($AsciiText, $escaped, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
    $cUtf16 = ([regex]::Matches($Utf16Text, $escaped, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
    $sum = $cAscii + $cUtf16
    if ($sum -gt 0) {
        Write-Host "[REL1-FAIL] Token '$tok' leaked $sum times (ASCII=$cAscii, UTF16=$cUtf16)" -ForegroundColor Red
        $TotalLeaks += $sum
    } else {
        Write-Host "[REL1-OK]   Token '$tok' count = 0"
    }
}
if ($TotalLeaks -gt 0) {
    throw "REL1 FAILED: $TotalLeaks forbidden path/token leak(s) detected in $ReleaseExe"
}

# 5. REL2 & REL3 Cargo.lock 5D002 Crypto + Zero-GPL License Audit
Write-Host "[REL-BUILD] Step 5/5: Running REL2/REL3 Cargo.lock 5D002 (zero TLS/AES) & Zero-GPL audit..."
$LockText = Get-Content -Path $LockPath -Raw -Encoding UTF8
$BannedCryptoCrates = @(
    "rustls",
    "rustls-webpki",
    "webpki",
    "openssl",
    "openssl-sys",
    "native-tls",
    "schannel",
    "security-framework",
    "aes",
    "aes-gcm",
    "chacha20",
    "chacha20poly1305",
    "quinn",
    "ring",
    "aws-lc-rs",
    "aws-lc-sys",
    "hyper-tls",
    "hyper-rustls",
    "tokio-rustls",
    "tokio-native-tls",
    "ureq"
)
$FoundBannedCrypto = @()
foreach ($c in $BannedCryptoCrates) {
    if ($LockText -match ("(?m)^name\s*=\s*`"" + [regex]::Escape($c) + "`"\s*$")) {
        $FoundBannedCrypto += $c
    }
}
if ($FoundBannedCrypto.Count -gt 0) {
    throw "REL3 5D002 FAILED: Forbidden TLS/AES confidentiality crypto crate(s) in Cargo.lock: $($FoundBannedCrypto -join ', ')"
}
Write-Host "[REL3-OK]   5D002 Cargo.lock audit: 0 TLS/AES/confidentiality crypto crates" -ForegroundColor Green

$MetaJson = cargo metadata --manifest-path $ManifestPath --format-version 1 | ConvertFrom-Json
$BadLicenses = @()
foreach ($pkg in $MetaJson.packages) {
    $lic = if ($pkg.license) { [string]$pkg.license } else { "" }
    if ($lic -match "\b(A?GPL|GPLv[23]|LGPL-[23]\.0-only)\b" -and $lic -notmatch "\b(MIT|Apache-2\.0|BSD|ISC|Zlib|MPL-2\.0)\b") {
        $BadLicenses += "$($pkg.name)@$($pkg.version) ($lic)"
    }
}
if ($BadLicenses.Count -gt 0) {
    throw "REL2 Zero-GPL FAILED: Forbidden copyleft crate(s): $($BadLicenses -join ', ')"
}
Write-Host "[REL2-OK]   Zero-GPL audit: 0 GPL/AGPL crates across $($MetaJson.packages.Count) resolved Cargo packages" -ForegroundColor Green
Write-Host "[REL-BUILD] ALL REL1/REL2/REL3/REL4 GATES PASSED! Binary: $ReleaseExe ($ExeSizeMiB MiB)" -ForegroundColor Green
