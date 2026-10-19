@echo off
REM Run StoreCounter automatically when the user logs in to Windows.
set "EXE=%~dp0StoreCounter.exe"
set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\StoreCounter.lnk"
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%LNK%'); $s.TargetPath='%EXE%'; $s.WorkingDirectory='%~dp0'; $s.Save()"
echo Autostart installed: %LNK%
pause
