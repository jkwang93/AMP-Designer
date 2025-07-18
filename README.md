# Discovery of novel antimicrobial peptides with notable antibacterial potency by a LLM-based foundation model

<div align=center>
<img src="./workflow.png" width="80%" height="80%" alt="TOC" align=center />
</div>

### Python Dependencies
```
Python >= 3.8
Pytorch >= 1.13.1
RDKit >= 2022.09.1
transformers >= 4.24.0
pandas >= 1.5.3
scipy == 1.10.0
```

## Overview
The train_AMP_GPT.py is the training code for pretrained model, train_prompt_contrast.py is the training code for AMP-Prompt model, train_distilation.py is the training code for AMP-Distillation model and the AMP-MIC/AMP-MIC.py is the code for predict MIC.

## Pretrained Model
The weight and configuration files of pre-trained model can be found in [Zenedo](https://zenodo.org/records/15012824).

## Generation
AMP_GPT_generator.py can be used for generation. 

## Cite
If you find this work interesting, please cite:
```
@article{wang2025discovery,
  title={Discovery of antimicrobial peptides with notable antibacterial potency by an LLM-based foundation model},
  author={Wang, Jike and Feng, Jianwen and Kang, Yu and Pan, Peichen and Ge, Jingxuan and Wang, Yan and Wang, Mingyang and Wu, Zhenxing and Zhang, Xingcai and Yu, Jiameng and others},
  journal={Science Advances},
  volume={11},
  number={10},
  pages={eads8932},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
```
