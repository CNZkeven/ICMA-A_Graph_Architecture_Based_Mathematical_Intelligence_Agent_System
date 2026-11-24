@echo off
cd /d E:\test\AI-MO\database\v2
rem Run the env's python directly - no "conda run" (it breaks log streaming).
rem Progress is written to build.log by build_db.py itself (FileHandler).
"F:\anac\envs\competition\python.exe" -u build_db.py
