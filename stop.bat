@echo off
echo FXダッシュボードを停止中...
taskkill /f /im streamlit.exe 2>nul
taskkill /f /fi "WINDOWTITLE eq FX Dashboard" 2>nul
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "PID"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr /i "streamlit" >nul && taskkill /f /pid %%a 2>nul
)
echo 停止しました。
pause
