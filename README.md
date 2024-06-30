<h2 align="center"> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_Self-Supervised_Multi-Object_Tracking_with_Path_Consistency_CVPR_2024_paper.pdf">Self-Supervised Multi-Object Tracking with Path Consistency</a></h2>

We propose a novel concept of path consistency to learn robust object matching without using manual object identity supervision. Our key idea is that, to track a object through frames, we can obtain multiple different association results from a model by varying the frames it can observe, i.e., skipping frames in observation. As the differences in observations do not alter the identities of objects, the obtained association results should be consistent. Based on this rationale, we formulate new Path Consistency Loss and have achieved SOTA on three tracking datasets (MOT17, PersonPath22, KITTI).

![image](assets/teaser.jpg)

## Preparation
### Install dependencies
```shell
pip3 install -r requirements.txt
cd libs
git clone https://github.com/ifzhang/ByteTrack.git
git clone https://github.com/JonathonLuiten/TrackEval.git
```

### Preprocess data

### MOT17
- download [MOT17 data](https://motchallenge.net/data/MOT17.zip), unzip it and place it as `data/mot17`
- download Tracktor++ detection from [drive](https://drive.google.com/file/d/179RgC8vidky7naAQc8Zuj2fIfeF17CsZ/view?usp=sharing) and place it as `data/mot17/tracktor_det`
- preprocess MOT17 data via `python -m src.script.preprocesss_mot17`

### PersonPath
- download [PersonPath22 dataset](https://amazon-science.github.io/tracking-dataset/personpath22.html).
- use `src/script/extract_pp_frames.py` to extract frames and place them in `data/personpath22/frames`.
- preprocess PersonPath22 data via `python -m src.script.preprocess_personpath.py`.

## Train model
```bash
python -m src.train --cfg src/configs/mot.yaml --set aux.gpu $GPUID aux.runid 1 # train MOT17
python -m src.train --cfg src/configs/personpath_fcos.yaml --set aux.gpu $GPUID aux.runid 1 # train PersonPath22
```
For example, the first command will train model on MOT17 public detections. Following prior works, the detections are refined by Tracktor++.
Results will be saved under `log`, where `ckpts` folder stores model weights. `saves` folder stores the accuracy of learned matching probability. Losses and accuracies are also visualized via wandb.

## Evaluate model
```bash
python -m src.inference_mot --gpu $GPUID --exp log/mot17/public/all/mot/1/ # eval MOT17
python -m src.inference_personpath --gpu $GPUID --exp log/personpath22/fcos_processed/all/personpath_fcos-replicate/1/ # eval PersonPath22
```
The first command will evaluate model on MOT17 videos, save results to `output/MOT17`, and print the accuracy for training videos. Accuracy for test videos can be obtained by submitting to MOT17 server.

The second command will evaluate model on PersonPath22 test videos.

## Pretrained Model
Pretrained model and results for MOT17 and PersonPath22 can be downloaded from [drive](https://drive.google.com/drive/folders/1zJyaDN0uTaDBYLVhnaA6JZyD7rNL4QWz?usp=sharing). Each zip file contains `log` and `output` folders. The performance are:

| MOT          | MOTA | IDF1 | IDSW |
|--------------|------|------|------|
| Train Videos | 64.8 | 69.0 | 542  |
| Test Videos  | 58.8 | 61.0 | 1162 |

| PersonPath22 | MOTA | IDF1 | IDSW |
|--------------|------|------|------|
| Test Videos  | 64.9 | 61.0 | 5234 |

## In Progress
The training configs for ~~PersonPath~~ and KITTI are exactly the same as defined in `src/configs/mot.yaml` except we set `pcl.G=20`. Unfortunately, we lost the data and model weights in a disk failure and need some time to replicate them. 

- [x] update training setting for Personpath.
- [ ] update training setting for KITTI.

## Citation
```text
@InProceedings{Lu_2024_CVPR,
    author    = {Lu, Zijia and Shuai, Bing and Chen, Yanbei and Xu, Zhenlin and Modolo, Davide},
    title     = {Self-Supervised Multi-Object Tracking with Path Consistency},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {19016-19026}
}
```



