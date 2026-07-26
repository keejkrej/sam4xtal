@echo off
REM Forward args so `run.cmd --mock` works.
powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
