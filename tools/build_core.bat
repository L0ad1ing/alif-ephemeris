@echo off
rem Build the optional compiled helpers on Windows with the Visual Studio Build Tools (x64):  tools\build_core.bat
rem The result, src\alif_ephemeris\_core.cp312-win_amd64.pyd, is picked up automatically; without it the library
rem uses the plain Python code that gives the same numbers.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
for /f "delims=" %%i in ('python -c "import sysconfig;print(sysconfig.get_paths()['include'])"') do set PYINC=%%i
for /f "delims=" %%i in ('python -c "import sys,os;print(os.path.join(sys.base_prefix,'libs'))"') do set PYLIBS=%%i
for /f "delims=" %%i in ('python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set EXT=%%i
cl /nologo /O2 /LD /I"%PYINC%" csrc\alif_core.c /Fe:src\alif_ephemeris\_core%EXT% /link /LIBPATH:"%PYLIBS%" /OUT:src\alif_ephemeris\_core%EXT%
del alif_core.obj 2>nul
del src\alif_ephemeris\_core%EXT:~0,-4%.exp src\alif_ephemeris\_core%EXT:~0,-4%.lib 2>nul
