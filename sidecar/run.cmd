@echo off
REM Convenience wrapper so `pnpm sidecar` / README work on Windows via Git Bash or this .cmd
powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1"
