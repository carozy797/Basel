
<p align="center">
<table>
  <tr>
    <td align="center" width="50%">
      <!-- <img src="https://github.com/jerryyangli.png" width="120" height="120" style="border-radius:50%;"><br><br> -->
      <b>Dr. Yang Li</b><br>
      <i>Department of Computer Science, Iowa State University</i><br>
      <a href="https://jerryyangli.github.io/">Website</a>
    </td>
    <td align="center" width="50%">
      <!-- <img src="https://github.com/carozy797.png" width="120" height="120" style="border-radius:50%;"><br><br> -->
      <b>Daniel Agyei Asante</b><br>
      <i>Department of Computer Science, 
        Iowa State University</i><br>
      <a href="https://carozy797.github.io/">Website</a>
    </td>
  </tr>
</table>
</p>

## 📖 Streamlining Language Models via Semantic Basis Analysis
### 📃 <a href="https://arxiv.org/pdf/2405.15877" target="_blank">Paper</a>

**Basel** is a principled low-rank compression framework designed to operate directly on the semantic structure of large language model weight matrices. It identifies the bases that encode high-impact semantic features for the target task and removes those with negligible contribution. This reduces weight parameters and memory footprint and improves inference throughput while preserving task accuracy. Basel achieves up to 2.7× model size reduction compared to state-of-the-art techniques and enables efficient deployment of language models on edge devices and in cost-sensitive environments.
Basel is validated across mathematical reasoning (GSM8K, MATH), code generation (HumanEval, MBPP), and on language modeling (WikiText-2).
Basel plays well with other compression methods and often beats them at their own game.

🔸 Basel with 8-bit achieves better accuracy than 4-bit quantization

🔸 Basel outperforms many existing low-rank and pruning-based compression methods

🔸 Models compressed with Basel remain stable under deeper compression levels where pruning accuracy collapses.

![](img/basel_overview.jpeg)


## 🔍 Table of Contents
- [🖥️ Software Dependencies](#software_dep)
- [🧩 Basel Part 1](#part1)
- [🚀 Basel Part 2](#part2)
- [💪 What Basel Delivers](#results)
- [📝 Citation](#citation)


<a id="software_dep"></a>
## 🖥️  Software Dependencies

### 1) Get the repo
```
git clone https://github.com/Iowa-State-University-AI-System-Group/Basel.git
cd Basel
```
### 2) Create & activate a virtual environment
### 3) Install PyTorch
```
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124
```
### 4) Install the project requirements
```
pip install -r requirements.txt
```

<a id="part1"></a>
## 🧩 Basel Part 1: Compression

Part 1 takes a dense model as input and generates a low-rank factorized model together with a dimension file specifying the shapes of its weight matrices.

<!-- ![](img/part1.png) -->
**Compression code**   [`train_bs_part1.py`](./train_bs_math_p1.py)


<a id="part2"></a>
## 🚀  Basel Part 2: Finetuning + Decompression

Part 2 takes the factorized model and dimension file as input, fine-tunes the factorized model, and decompresses it into an equivalent dense model. The decompressed dense model is generated solely to facilitate convenient performance evaluation of the fine-tuned low-rank model.

**Fine-tuning + decompression code**   [`train_bs_part2.py`](./train_bs_math_p2.py)


<a id="results"></a>
## 💪 What Basel Delivers
### 1. Mathematical Reasoning Task
![](img/basel_math_7b.jpeg)
### 2. Programming Task
![](img/basel_7b_prog.jpeg)
### 3. Language Modeling Task
![](img/basel_ppl.jpeg)

<a id="citation"></a>
## 📝 Citation
```
@article{basel,
      title={{Streamlining Language Models via Semantic Basis Analysis}}, 
      author={Li, Yang and Asante, Daniel Agyei and Zhao, Changsheng and Chang, Ernie and Shi, Yangyang and Chandra, Vikas},
      journal={Transactions on Machine Learning Research}, 
      year={2025},
}

@article{basis_selection,
      title={{Basis Selection: Low-Rank Decomposition of Pretrained Large Language Models for Target Applications}}, 
      author={Yang Li and Daniel Agyei Asante and Changsheng Zhao and Ernie Chang and Yangyang Shi and Vikas Chandra},
      year={2024},
      journal={arxiv preprint arXiv:2405.15877}, 
}
```
</p>
 
