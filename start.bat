@echo off
cd /d %~dp0
pipenv run streamlit run app.py
pause
