@echo off
rem Builds this branch for AMD GPUs with the HIP SDK for Windows (measured with HIP SDK 7.2 on gfx1100).
rem Usage: scripts\inkling2-amd\build-hip.cmd [build-dir]   (default build-hip, relative to the repository root)
rem Needs the HIP SDK for Windows (HIP_PATH is set by its installer) and the Visual Studio C++ build tools.
rem Optional environment variables:
rem   GPU_TARGETS  GPU architectures to compile for, e.g. gfx1100 or gfx1100;gfx1101 (default gfx1100)
rem   VCVARS       path of vcvars64.bat (default: the latest Visual Studio with C++ tools, found with vswhere)
rem   INK_CMAKE    extra cmake options at configure time, e.g. -DLLAMA_BUILD_TESTS=ON
rem   INK_TARGETS  build targets (default: llama-server llama-bench llama-perplexity llama-cli llama-quantize)
rem The configure step runs only once per build directory (delete it to change the options).
setlocal
if not defined HIP_PATH goto no_hip
if not defined GPU_TARGETS set "GPU_TARGETS=gfx1100"
if not defined INK_TARGETS set "INK_TARGETS=llama-server llama-bench llama-perplexity llama-cli llama-quantize"
if defined VCVARS goto have_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto no_vs
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto no_vs
set "VCVARS=%VSDIR%\VC\Auxiliary\Build\vcvars64.bat"
:have_vcvars
if not exist "%VCVARS%" goto no_vs
call "%VCVARS%" >nul || exit /b 1
rem CMake and Ninja shipped with Visual Studio
set "PATH=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "BD=%~1"
if "%BD%"=="" set "BD=build-hip"
cd /d "%~dp0..\.." || exit /b 1
if exist "%BD%\CMakeCache.txt" goto build
cmake -S . -B "%BD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="%HIP_PATH%bin\clang.exe" -DCMAKE_CXX_COMPILER="%HIP_PATH%bin\clang++.exe" -DCMAKE_PREFIX_PATH="%HIP_PATH:~0,-1%" -DGGML_HIP=ON -DGPU_TARGETS="%GPU_TARGETS%" -DCMAKE_HIP_ARCHITECTURES="%GPU_TARGETS%" -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF %INK_CMAKE% || exit /b 1
:build
cmake --build "%BD%" --target %INK_TARGETS% -j %NUMBER_OF_PROCESSORS% || exit /b 1
echo BUILD OK: %BD%\bin
exit /b 0
:no_hip
echo HIP_PATH is not set: install the HIP SDK for Windows first
exit /b 1
:no_vs
echo Visual Studio C++ build tools not found: install them, or set VCVARS to the path of vcvars64.bat
exit /b 1
