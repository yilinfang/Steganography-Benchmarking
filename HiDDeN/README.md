# HiDDeN 

## 1) Install
```powershell
cd C:\Users\shenz\HiDDeN\hidden_demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy pillow opencv-python pandas tqdm matplotlib
```


## 2) Checkpoint\ (.pyt) comes from

checkpoints/coco2017-hidden-224-e20--epoch-20.pyt is a trained HiDDeN checkpoint produced by our training run on COCO2017, using 224×224 images and max message length L=30 bits (original HiDDeN design). We copy this trained checkpoint into hidden_demo/checkpoints/ for standalone benchmarking.

## 3) Run (export CSV)

### Run all benchmarks (one command)

From the `HiDDeN/` folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

This will execute all benchmark modes and save CSV outputs to hidden_demo/results/.


### Run benchmarks (other ways)
```powershell
#Payload mode
python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode payload `
  --csv-out .\results\payload.csv


#Gaussian noise
python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode gauss `
  --csv-out .\results\gauss_sigma010.csv `


#Gaussian blur
python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode blur `
  --csv-out .\results\blur_r3.csv `


#Rotation
python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode rotate `
  --csv-out .\results\rotate_5deg.csv `


#JPEG compression
python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode jpeg `
  --csv-out .\results\jpeg_q50.csv `


#Brightness / Contrast
python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode brightness `
  --csv-out .\results\brightness_12.csv `


python .\benchmark_hidden.py `
  --checkpoint .\checkpoints\coco2017-hidden-224-e20--epoch-20.pyt `
  --data-dir .\data\demo_images `
  --mode contrast `
  --csv-out .\results\contrast_08.csv `
```

## 6) Outputs

All runs export CSV files into results/, containing:
1. recovery rate / BER (depending on mode)
2. optional timing (encoding/decoding), if enabled in the script
3. payload configuration in payload mode
