$csv = "data\data\handbag\data1\raw\imu1.csv"
$weights = "tcn_imu_weights.json"

$pyOut = "pytorch_reference.csv"
$cppOut = "cpp_output.csv"

python infer.py --csv $csv --weights $weights --out $pyOut

g++ infer.cpp -O2 -std=c++17 -o infer.exe

.\infer.exe $csv $weights $cppOut

python compare_outputs.py --ref $pyOut --cpp $cppOut