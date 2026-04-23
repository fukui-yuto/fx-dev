@echo off
cd /d %~dp0
echo FXダッシュボードをバックグラウンドで起動中...
start "FX Dashboard" /min cmd /c "pipenv run streamlit run app.py"
echo 起動しました。このウィンドウを閉じてもダッシュボードは動き続けます。
echo 停止するには: タスクマネージャーから streamlit / python を終了してください。
echo または stop.bat を実行してください。
timeout /t 3
