# HealthiVert-GAN: Pseudo-Healthy Vertebral Image Synthesis for Interpretable Compression Fracture Grading

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
[![PyTorch Docs](https://pytorch.org/)](https://pytorch.org/)

**HealthiVert-GAN** is a novel framework for synthesizing pseudo-healthy vertebral CT images from fractured vertebrae. By simulating pre-fracture states, it enables interpretable quantification of vertebral compression fractures (VCFs) through **Relative Height Loss of Vertebrae (RHLV)**. The model integrates a two-stage GAN architecture with anatomical consistency modules, achieving state-of-the-art performance on both public and private datasets.

---

## 🚀 Key Features
- **Two-Stage Synthesis**: Coarse-to-fine generation with 2.5D sagittal/coronal fusion.
- **Anatomic Modules**:
  - **Edge-Enhancing Module (EEM)**: Captures precise vertebral morphology.
  - **Self-adaptive Height Restoration Module (SHRM)**: Predicts healthy vertebral height adaptively.
  - **HealthiVert-Guided Attention Module (HGAM)**: Focuses on non-fractured regions via Grad-CAM++.
- **Iterative Synthesis**: Generates adjacent vertebrae first to minimize fracture interference.
- **RHLV Quantification**: Measures height loss in anterior/middle/posterior regions for SVM-based Genant grading.

---

## 🛠️ Architecture

![Workflow](images/workflow.png)

### Workflow
1. **Preprocessing**: 
   - **Spine Straightening**: Align vertebrae vertically using SCNet segmentation.
   - **De-pedicle**: Remove vertebral arches for body-focused analysis.
   - **Masking**: Replace target vertebra with a fixed-height mask (40mm).
   
2. **Two-Stage Generation**:
   - **Coarse Generator**: Outputs initial CT and segments adjacent vertebrae.
   - **Refinement Generator**: Enhances details with contextual attention and edge loss.

3. **Iterative Synthesis**:
   - Step 1: Synthesize adjacent vertebrae.
   - Step 2: Generate target vertebra using Step 1 results.

4. **RHLV Calculation**:
   ```math
   RHLV = \frac{H_{syn} - H_{ori}}{H_{syn}}
   ```
   Segments vertebra into anterior/middle/posterior regions for detailed analysis.

   **SVM Classification**: Uses RHLV values to classify fractures into mild/moderate/severe.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/HealthiVert-GAN.git
cd HealthiVert-GAN
pip install -r requirements.txt  # PyTorch, NiBabel, SimpleITK, OpenCV
```

### Data Preparation

#### Dataset Structure
Organize data as:

```
/dataset/
  ├── patient_001/
  │   ├── patient_001_ct.nii.gz    # Original CT
  │   └── patient_001_seg.nii.gz   # Vertebrae segmentation
  └── patient_002/
      ├── patient_002_ct.nii.gz
      └── patient_002_seg.nii.gz
```

#### Preprocessing

**Spine Straightening**:

```bash
python straighten/location_json_local.py  # Generate vertebral centroids
python straighten/straighten_mask_3d.py   # Output: ./dataset/straightened/
```

**Attention Map Generation**:

```bash
python Attention/grad_CAM_3d_sagittal.py  # Output: ./Attention/heatmap/
```

### Training

**Configure JSON**:

Update `vertebra_data.json` with patient IDs, labels, and paths.

**Train Model**:

```bash
python train.py \
  --dataroot ./dataset/straightened \
  --name HealthiVert_experiment \
  --model pix2pix \
  --direction BtoA \
  --batch_size 16 \
  --n_epochs 1000
```

Checkpoints saved in `./checkpoints/HealthiVert_experiment`.

### Inference

**Generate Pseudo-Healthy Vertebrae**:

```bash
python eval_3d_sagittal_twostage.py \
  --dataroot ./dataset/straightened \
  --name HealthiVert_experiment \
  --model test
```

Outputs: `./output/CT_fake/` and `./output/label_fake/`.

**Fracture Grading**

**Calculate RHLV**:

```bash
python RHLV_calculation.py --input_dir ./output/CT_fake
```

**Train SVM Classifier**:

```bash
python SVM_classifier.py --data_path ./results/RHLV_metrics.csv
```

---

## 📊 Results

### Qualitative Comparison

| Method              | Synthetic Vertebra | Height Loss Heatmap |
|---------------------|--------------------|---------------------|
| Traditional (pix2pix) | pix2pix            | N/A                 |
| HealthiVert-GAN     | Ours               | Heatmap            |

### Quantitative Performance (Verse2019 Dataset)

| Metric      | HealthiVert-GAN | AOT-GAN [32] | 3D CNN [23] |
|-------------|-----------------|--------------|-------------|
| SSIM        | 0.92            | 0.88         | 0.85        |
| RHDR (%)    | 4.3             | 6.1          | 8.7         |
| Macro-F1    | 0.88            | 0.82         | 0.76        |

---

## 📜 Citation

```bibtex
@article{zhang2024healthivert,
  title={HealthiVert-GAN: A Novel Framework of Pseudo-Healthy Vertebral Image Synthesis for Interpretable Compression Fracture Grading},
  author={Zhang, Qi and Zhang, Shunan and Zhao, Ziqi and Wang, Kun and Xu, Jun and Sun, Jianqi},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2024}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
