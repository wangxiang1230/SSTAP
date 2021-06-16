# SSTAP
Pytorch implementation of the paper: "Self-Supervised Learning for Semi-Supervised Temporal Action Proposal" (CVPR-2021)

## Requirements
The code runs correctly with:

* python 3.8.5
* pytorch 1.6.0
* torchvision 0.7.0

Other versions may also work. 

## Feature and model weights coming soon...

## Prepare 
Generate labeled/unlabeled data (you can also use our files directly)
```
python gen_unlabel_videos.py
```

## Training and Validation
```
bash SSTAP.sh | tee log_SSTAP.txt
```

## Acknowledgement

[BMN: Boundary-Matching Network](https://github.com/JJBOY/BMN-Boundary-Matching-Network) 

[TSN-Feature](https://github.com/wzmsltw/BSN-boundary-sensitive-network)

## Citation
If our code is helpful for your reseach, please cite our paper:

```
@inproceedings{SSTAP,
  title={Self-Supervised Learning for Semi-Supervised Temporal Action Proposal},
  author={Wang, Xiang and Zhang, Shiwei and Qing, Zhiwu and Shao, Yuanjie and Gao, Changxin and Sang, Nong},
  booktitle={CVPR},
  year={2021}
}
```
