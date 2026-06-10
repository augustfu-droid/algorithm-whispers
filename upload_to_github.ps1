param(
  [string]$Repository = "augustfu-droid/algorithm-whispers",
  [string]$Branch = "main",
  [string]$Message = "docs: revise sources and append PDF errata"
)

$ErrorActionPreference = "Stop"

if (-not $env:GITHUB_TOKEN) {
  throw "Set GITHUB_TOKEN to a token with contents:write permission before running this script."
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Files = @(
  "README.md",
  "CHANGELOG.md",
  "REVIEW_NOTES.md",
  "article.md",
  "mathvol.md",
  "article.pdf",
  "mathvol.pdf",
  "upload_to_github.ps1"
)

$Headers = @{
  "Authorization" = "Bearer $env:GITHUB_TOKEN"
  "Accept" = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
  "User-Agent" = "algorithm-whispers-upload"
}

function Invoke-GitHubJson {
  param(
    [string]$Method,
    [string]$Uri,
    [object]$Body = $null
  )
  if ($null -eq $Body) {
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers
  }
  $Json = $Body | ConvertTo-Json -Depth 20 -Compress
  $Utf8Body = [System.Text.Encoding]::UTF8.GetBytes($Json)
  return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers -Body $Utf8Body -ContentType "application/json; charset=utf-8"
}

$Api = "https://api.github.com/repos/$Repository"
$BranchInfo = Invoke-GitHubJson -Method Get -Uri "$Api/branches/$Branch"
$ParentSha = $BranchInfo.commit.sha
$BaseTreeSha = $BranchInfo.commit.commit.tree.sha

$Tree = @()
foreach ($File in $Files) {
  $Path = Join-Path $RepoRoot $File
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "Missing file: $Path"
  }

  $Bytes = [System.IO.File]::ReadAllBytes($Path)
  $Base64 = [Convert]::ToBase64String($Bytes)
  $Blob = Invoke-GitHubJson -Method Post -Uri "$Api/git/blobs" -Body @{
    content = $Base64
    encoding = "base64"
  }

  $Tree += @{
    path = $File
    mode = "100644"
    type = "blob"
    sha = $Blob.sha
  }
}

$NewTree = Invoke-GitHubJson -Method Post -Uri "$Api/git/trees" -Body @{
  base_tree = $BaseTreeSha
  tree = $Tree
}

$Commit = Invoke-GitHubJson -Method Post -Uri "$Api/git/commits" -Body @{
  message = $Message
  tree = $NewTree.sha
  parents = @($ParentSha)
}

Invoke-GitHubJson -Method Patch -Uri "$Api/git/refs/heads/$Branch" -Body @{
  sha = $Commit.sha
  force = $false
} | Out-Null

Write-Host "Uploaded commit: $($Commit.sha)"
Write-Host "https://github.com/$Repository/commit/$($Commit.sha)"
