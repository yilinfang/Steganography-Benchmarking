$ckpt = ".\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt"
$data = "..\data\demo_images"
$out  = ".\results"

New-Item -ItemType Directory -Force -Path $out | Out-Null

python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode payload     --csv-out "$out\payload.csv"
python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode gauss       --csv-out "$out\gauss.csv"
python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode blur        --csv-out "$out\blur.csv"
python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode rotate      --csv-out "$out\rotate.csv"
python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode jpeg        --csv-out "$out\jpeg.csv"
python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode brightness  --csv-out "$out\brightness.csv"
python .\benchmark_hidden.py --checkpoint $ckpt --data-dir $data --mode contrast    --csv-out "$out\contrast.csv"

Write-Host "Done. CSVs saved to $out"
