
$env:HSLLIB_PATH = "c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\Libraries\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin\libcoinhsl.dll"
$env:PATH = "c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\Libraries\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin;" + 
            "C:\Users\maxed\miniconda3\envs\larrak;" +
            "C:\Users\maxed\miniconda3\envs\larrak\Library\mingw-w64\bin;" +
            "C:\Users\maxed\miniconda3\envs\larrak\Library\usr\bin;" +
            "C:\Users\maxed\miniconda3\envs\larrak\Library\bin;" +
            "C:\Users\maxed\miniconda3\envs\larrak\Scripts;" + 
            $env:PATH

Write-Host "Checking MA86 Availability..."
C:\Users\maxed\miniconda3\envs\larrak\python.exe tests/infra/check_ma86.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "MA86 Check Passed. Running DOE..."
    C:\Users\maxed\miniconda3\envs\larrak\python.exe tests/goldens/phase1/generate_doe.py
} else {
    Write-Host "MA86 Check Failed."
    exit 1
}
