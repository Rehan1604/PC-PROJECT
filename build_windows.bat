@echo off
REM build_windows.bat - native Windows build (if you don't want to use WSL)
REM PC project ODD-2026 - Rehan Mehta (241080039), Bhavya Gothi (241080021)
REM
REM Needs MSYS2 with the mingw-w64 g++ on PATH  (pacman -S mingw-w64-x86_64-gcc)
REM For the MPI builds also install Microsoft MPI (msmpisetup.exe + msmpisdk.msi),
REM which sets MSMPI_INC and MSMPI_LIB64.  Run them with:  mpiexec -n 4 bin\flood_mpi.exe --balance

if not exist bin mkdir bin
set FLAGS=-O3 -march=native -std=c++17 -ffp-contract=off -fno-math-errno -fopenmp-simd

echo building serial...
g++ %FLAGS% src\flood.cpp -o bin\flood_serial.exe || goto :fail
echo building openmp...
g++ %FLAGS% -fopenmp src\flood.cpp -o bin\flood_omp.exe || goto :fail
echo building no-simd baseline...
g++ -O3 -march=native -std=c++17 -ffp-contract=off -fno-math-errno -fno-tree-vectorize -DSCALAR_CBRT src\flood.cpp -o bin\flood_novec.exe || goto :fail

if "%MSMPI_INC%"=="" (
    echo MS-MPI SDK not found, skipping the MPI builds
    goto :done
)
echo building mpi...
g++ %FLAGS% -DUSE_MPI -DMSMPI_NO_SAL -I"%MSMPI_INC%" -I"%MSMPI_INC%\x64" src\flood.cpp -L"%MSMPI_LIB64%" -lmsmpi -o bin\flood_mpi.exe || goto :fail
echo building hybrid...
g++ %FLAGS% -fopenmp -DUSE_MPI -DMSMPI_NO_SAL -I"%MSMPI_INC%" -I"%MSMPI_INC%\x64" src\flood.cpp -L"%MSMPI_LIB64%" -lmsmpi -o bin\flood_hybrid.exe || goto :fail

:done
echo done, binaries are in bin\
exit /b 0

:fail
echo build failed
exit /b 1
