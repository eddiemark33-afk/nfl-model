# Weekly routine: fit ratings, print this week's card, back it up to GitHub.
# Run Tuesday after Monday Night Football, once nflverse has posted the week's lines.
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python run_week.py
git add .
git commit -m "weekly card"
git push
Write-Output "DONE"
