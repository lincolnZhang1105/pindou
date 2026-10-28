@echo off
rem Run the given command inside an MSVC x64 developer environment (Windows local dev).
rem Usage: tools\msvc.cmd cmake --build build/windows
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo [msvc] Visual Studio with C++ tools not found 1>&2
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
%*
