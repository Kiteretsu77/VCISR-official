# VCISR: Blind Single Image Super-Resolution with Video Compression Synthetic Data (WACV 2024)
> [![paper](https://img.shields.io/badge/Paper-<COLOR>.svg)](https://openaccess.thecvf.com/content/WACV2024/papers/Wang_VCISR_Blind_Single_Image_Super-Resolution_With_Video_Compression_Synthetic_Data_WACV_2024_paper.pdf)<br>
:star:If you like VCISR, please help star this repo. Thanks!:hugs:

## :book:Table Of Contents
- [Update](#update)
- [Installation](#installation)
- [Train](#train)
- [Inference](#inference)
- [Anime](#Anime)
- [VQ-RealLQ](#VQ-RealLQ)

## <a name="update"></a>Update
- **2023.12.08**: This pre-trained weight is released.
- **2023.11.29**: This repo is released.


## <a name="installation"></a> Installation (Environment Preparation)

```shell
git clone git@github.com:Kiteretsu77/VCISR-official.git
cd VCISR

# Create conda env
conda create -n VCISR python=3.10
conda activate VCISR

# Install Pytorch we use torch.compile in our repository by default
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install FFMPEG (the following is for linux system, the rest can see https://ffmpeg.org/download.html)
sudo apt install ffmpeg
```



## <a name="train"></a> Train
1. Download Datasets (DIV2K) and crop them by the script below (following our paper):

    ```shell
    bash scripts/download_datasets.sh
    ```

2. Train: Please check **opt.py** to setup parameters you want\
    **Step1** (Net L1 loss training): Run 
    ```shell
    python train_code/train.py 
    ```
    The model weights will be inside the folder 'saved_models'

    **Step2** (GAN Adversarial Training): 
    1. Change opt['architecture'] in **opt.py** as "GRLGAN".
    2. Rename weights in 'saved_models' (either closest or the best, we use closest weight) to **grlgan_pretrained.pth**
    3. Run 
    ```shell
    python train_code/train.py --use_pretrained
    ```



## <a name="inference"></a> Inference:
1. Download the weight from https://drive.google.com/file/d/1Mbrw1ji_qcOteuSOkZqVgSEda_PQ40tA/view?usp=drive_link  and put them in "saved_models" folder
2. **Setup the configuration of test_code/inference.py after line 215**. 
2. Then, Execute 
    ```shell
    python test_code/inference.py
    ```

## <a name="Anime"></a> Anime:
We also extend our methods on the Anime Restoration and Super-Resolution task with public and private Anime datasets. \
You can also find a pre-built **highly accelerated** Anime VSR inference repository from: https://github.com/Kiteretsu77/FAST_Anime_VSR. \
We highly recommend users to checkout this repository by myself if you want extreme high speed in inference.
In the future, we will release more details related to the Anime extension and versatile pre-trained weights.

## <a name="VQ-RealLQ"></a> VQ-RealLQ:
The small image inference dataset will be released soon. If you need it earlier, you can contact: hikaridawn412316@gmail.com.



## Citation
Please cite us if our work is useful for your research.
```
@article{wang2023vcisr,
  title={VCISR: Blind Single Image Super-Resolution with Video Compression Synthetic Data},
  author={Wang, Boyang and Liu, Bowen and Liu, Shiyu and Yang, Fengyu},
  journal={arXiv preprint arXiv:2311.00996},
  year={2023}
}
```


## License
This project is released under the [GPL 3.0 license](LICENSE).

## Contact
If you have any questions, please feel free to contact with me at hikaridawn412316@gmail.com or boyangwa@umich.edu.

