# SSTAP
Pytorch implementation of the paper: "Self-Supervised Learning for Semi-Supervised Temporal Action Proposal" (CVPR-2021) 
[[SSTAP-Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Self-Supervised_Learning_for_Semi-Supervised_Temporal_Action_Proposal_CVPR_2021_paper.pdf)

## Requirements
The code runs correctly with:

* python 3.8.5
* pytorch 1.6.0
* torchvision 0.7.0

Other versions may also work. 

## Feature and model weights
* I3D Feature. [[here]](https://zenodo.org/record/5035205#.YNmAhLvitPY)
* TSN Feature. [[BSN]](https://github.com/wzmsltw/BSN-boundary-sensitive-network)
* Model Weights. [[here]](https://zenodo.org/record/5036065#.YNmAE7vitPY)
* SlowFast101 feature used by the winners of the [CVPR2020 ActivityNet Temporal Action Localization Challenge](https://arxiv.org/abs/2006.07526) and the [CVPR2021 ActivityNet Temporal Action Localization Challenge](https://arxiv.org/abs/2106.11812) is coming soon，and ActivityNet Challenge website is [here](http://activity-net.org/challenges/2021/tasks/anet_localization.html).

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
