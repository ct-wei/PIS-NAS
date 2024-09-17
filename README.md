# PIS-NAS: Pyramid Information System-Based Neural Architecture Search for Object Detection

## Introduction
This repository is the official implementation  of "PIS-NAS: Pyramid Information System-Based Neural Architecture Search for Object Detection". 

Chentian Wei<sup>1,2</sup>, [Ao Li](https://liaosite.github.io/)<sup>1</sup>, Lei Pu<sup>3</sup>, [Le Dong](https://faculty.xidian.edu.cn/DL4/zh_CN/index/430205/list/index.htm) <sup>1\*</sup>, [Weisheng Dong](https://see.xidian.edu.cn/faculty/wsdong/)<sup>1</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University

<sup>2</sup>Institute for Network Sciences and Cyberspace, Tsinghua University

<sup>3</sup>Combat support College, Rocket Force University of Engineering

*: Corresponding Author. Email: dongle@xidian.edu.cn

Code and Usage is now available.
## Usage
To use this repository, you need to follow these steps:

1. Clone the repository:
```
git clone https://github.com/ct-wei/PIS-NAS.git
```

2. Install the required packages:
```
pip install -r requirements/nas.txt
```

3. Search the model:
```
python tools/search.py configs/erf/cspnet/yolov8n_erf.py
```

## Citation
If you find this useful, please support us by citing them.
```
@inproceedings{,
	title = {PIS-NAS: Pyramid Information System Based Neural Architecture Search for Object Detection},
	author = {Wei, Chentian and Li, Ao and Pu, Lei and Dong, Le and Dong, Weisheng},
	booktitle = {},
	year = {},
	url = {}
}
```
