# P2PNet+IADM

P2PNet is a neural network framework that uses a point regression approach to address the problem of crowd density estimation. It is capable of not only counting the number of people but also providing the location information of individual heads.

The IADM module, introduced in RGBT-CC for dealing with RGBT data, effectively integrates RGB and T (thermal) data. The official code only incorporates the IADM module into BL and CSRNet, but both BL and CSRNet are based on density maps and cannot provide location information.

Therefore, I have integrated the IADM module into P2PNet for crowd density estimation on RGBT datasets.


### Installation
```
pip3 install -r requirements.txt
```

### Download dataset:
Download RGBT-CC Dataset & Models: [<a href="https://www.dropbox.com/sh/o4ww2f5tv3nay9n/AAA4CfVMTZcdwsFxFlhwDsSba?dl=0">Dropbox</a>][<a href="https://pan.baidu.com/s/1ui265kpRGIpTu9kLQrEYgA">BaiduYun (PW: RGBT)</a>]

### Preprocess:
```
# First change the data_root to your dataset root
python3 preprocess_RGBT.py
```

### Training:
```
python3 train.py --data_root you_rgbt_data_root
```

### Test:
```
python3 run_test.py
```

### Onnx:
```
python3 export.py --weight_path weights/best_mae.pth
```

### Citation:
[P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) and [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting)