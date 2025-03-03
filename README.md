# Project 1 Image Enhancement

Work by

![GitHub Copilo1t](https://img.shields.io/badge/Liu_Yifei-4285F4?style=for-the-badge&logo=github-copilot&logoColor=white)


To get the enhancement result by three methods, run
(of cause you should prepare the proper environment first)
```bash
PYTHONPATH=$PYTHONPATH:. python processed/gamma/gamma.py
PYTHONPATH=$PYTHONPATH:. python processed/LDR/ldr.py
PYTHONPATH=$PYTHONPATH:. python processed/MF/mf.py
```

For the NIQE metric, see `niqe-main/example.m`

For the "Structure-revealing low-light image enhancement via robust retinex model.", see `Low-light-image-enhancement-master/demo.m`

For all the images, see `processed` folder

```text
processed
├── gamma
│   ├── 1.png
│   ├── 497.png
│   ├── 748.png
│   ├── 776.png
│   ├── 780.png
│   ├── gamma.py
│   └── mae_results.txt
├── LDR
│   ├── enhanced_1.png
│   ├── enhanced_497.png
│   ├── enhanced_748.png
│   ├── enhanced_776.png
│   ├── enhanced_780.png
│   ├── ldr.py
│   └── metrics.txt
├── MF
│   ├── 1.png
│   ├── 497.png
│   ├── 748.png
│   ├── 776.png
│   ├── 780.png
│   ├── mae_results.txt
│   └── mf.py
└── robost
    ├── 1.png
    ├── 497.png
    ├── 748.png
    ├── 776.png
    └── 780.png
5 directories, 26 files
```