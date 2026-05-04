$csv = "data\data\handbag\data1\raw\imu2.csv"
$weights = "tcn_imu_weights.json"

$pyOut = "pytorch_reference.csv"
$cppOut = "cpp_output.csv"

Remove-Item $pyOut -ErrorAction SilentlyContinue
Remove-Item $cppOut -ErrorAction SilentlyContinue
Remove-Item infer.exe -ErrorAction SilentlyContinue

python infer.py --csv $csv --weights $weights --out $pyOut

g++ infer.cpp -O2 -std=c++17 -o infer.exe
if ($LASTEXITCODE -ne 0) { exit 1 }

.\infer.exe $csv $weights $cppOut
if ($LASTEXITCODE -ne 0) { exit 1 }

python compare_outputs.py --ref $pyOut --cpp $cppOut