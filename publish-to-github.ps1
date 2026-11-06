param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repository = "https://github.com/321222/DermaCare-AI-.git"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$gitCommand = Get-Command git -ErrorAction SilentlyContinue

if (-not $gitCommand) {
    $gitCandidates = @(
        (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Git\cmd\git.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
    )
    $desktopGit = Join-Path $env:LOCALAPPDATA "GitHubDesktop"
    if (Test-Path -LiteralPath $desktopGit) {
        $gitCandidates += Get-ChildItem -LiteralPath $desktopGit -Filter git.exe -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\resources\\app\\git\\cmd\\git\.exe$' } |
            Select-Object -ExpandProperty FullName
    }
    $gitPath = $gitCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $gitPath) {
        throw "Git is not installed or was not found. Install Git for Windows, reopen PowerShell, and run this script again."
    }
    $gitCommand = Get-Command $gitPath
}
$git = $gitCommand.Source

$copyFiles = @(
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "DEPLOYMENT-QUICKSTART.md",
    "Dockerfile",
    "README.md",
    "app.py",
    "audio_utils.py",
    "chatbot.py",
    "compose.yaml",
    "database.py",
    "evaluate.py",
    "gradcam.py",
    "knowledge.py",
    "publish-to-github.ps1",
    "rag.py",
    "report.py",
    "requirements.txt",
    "train.py",
    "triage.py"
)
$copyFolders = @(".github", "knowledge_base", "tests")

foreach ($file in $copyFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $file) -PathType Leaf)) {
        throw "Required project file is missing: $file"
    }
}

$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("dermacare-publish-" + [guid]::NewGuid().ToString("N"))
$clonePath = Join-Path $stageRoot "repository"
New-Item -ItemType Directory -Path $stageRoot | Out-Null
$completed = $false

try {
    Write-Host "Cloning $repository into a temporary staging folder..."
    & $git clone --branch main $repository $clonePath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not clone the target repository. Check network access and that its default branch is main."
    }

    foreach ($file in $copyFiles) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $file) -Destination (Join-Path $clonePath $file) -Force
    }

    foreach ($folder in $copyFolders) {
        $sourceFolder = Join-Path $projectRoot $folder
        $destinationFolder = Join-Path $clonePath $folder
        if (-not (Test-Path -LiteralPath $sourceFolder -PathType Container)) {
            throw "Required project folder is missing: $folder"
        }
        New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
        Get-ChildItem -LiteralPath $sourceFolder -File -Recurse |
            Where-Object {
                $_.Extension -in @(".py", ".yml", ".yaml", ".txt")
            } |
            ForEach-Object {
                $relative = $_.FullName.Substring($sourceFolder.Length).TrimStart("\")
                $destination = Join-Path $destinationFolder $relative
                $parent = Split-Path -Parent $destination
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
                Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
            }
    }

    foreach ($folder in @("dataset", "models")) {
        $sourceFolder = Join-Path $projectRoot $folder
        $destinationFolder = Join-Path $clonePath $folder
        New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
        foreach ($file in @("README.md", ".gitkeep")) {
            $sourceFile = Join-Path $sourceFolder $file
            if (Test-Path -LiteralPath $sourceFile -PathType Leaf) {
                Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $destinationFolder $file) -Force
            }
        }
    }

    $allowedPaths = @(
        ".dockerignore", ".env.example", ".gitignore", "DEPLOYMENT-QUICKSTART.md", "Dockerfile", "README.md",
        "app.py", "audio_utils.py", "chatbot.py", "compose.yaml", "database.py",
        "dataset", "evaluate.py", ".github", "gradcam.py", "knowledge.py",
        "knowledge_base", "models", "publish-to-github.ps1", "rag.py", "report.py",
        "requirements.txt", "tests", "train.py", "triage.py"
    )
    & $git -C $clonePath add -- $allowedPaths
    if ($LASTEXITCODE -ne 0) {
        throw "Git could not stage the approved project files."
    }

    Write-Host ""
    Write-Host "Staged change summary:"
    & $git -C $clonePath diff --cached --stat
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect staged changes."
    }

    $stagedNames = & $git -C $clonePath diff --cached --name-only
    if ($stagedNames | Where-Object {
        $leaf = Split-Path -Leaf $_
        (($leaf -like ".env*" -and $leaf -ne ".env.example") -or $leaf -like "*.env") -or
            $_ -match '(^|/)secrets\.toml$|\.sqlite3?$|\.keras$|\.(jpg|jpeg|png)$'
    }) {
        throw "A local secret, database, model, or image is staged. Refusing to publish."
    }

    if ($DryRun) {
        Write-Host "Dry run completed. No commit was created and nothing was pushed."
        $completed = $true
        return
    }

    if (-not $stagedNames) {
        Write-Host "No changes to publish."
        $completed = $true
        return
    }

    $approval = Read-Host "Commit these changes and push them to $repository? Type PUSH to confirm"
    if ($approval -cne "PUSH") {
        Write-Host "Cancelled. No commit was created and nothing was pushed."
        $completed = $true
        return
    }

    & $git -C $clonePath commit -m "Add DermaCare AI application and deployment stack" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    if ($LASTEXITCODE -ne 0) {
        throw "Git commit failed. Configure your Git author identity and rerun the script."
    }

    & $git -C $clonePath push origin HEAD:main
    if ($LASTEXITCODE -ne 0) {
        throw "Push failed. Authenticate with Git Credential Manager/GitHub Desktop and retry; no credentials should be entered into this script."
    }

    Write-Host "Published successfully to https://github.com/321222/DermaCare-AI-"
    $completed = $true
}
finally {
    if ($completed -and (Test-Path -LiteralPath $stageRoot)) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
    elseif (Test-Path -LiteralPath $stageRoot) {
        Write-Warning "Staging folder retained for troubleshooting: $stageRoot"
    }
}
