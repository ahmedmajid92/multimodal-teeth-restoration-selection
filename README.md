# Multimodal Classification System for Tooth Restoration Types

A comprehensive machine learning system for classifying dental restoration types (Direct vs. Indirect) using hybrid multimodal approaches combining tabular clinical data and dental radiographic images.

## 🎯 Overview

This system implements a state-of-the-art multimodal classification pipeline that integrates:
- **Clinical tabular features** (depth, width, lesion characteristics, patient demographics)
- **Dental radiographic images** (preprocessed and augmented X-ray/clinical photos)
- **Hybrid ensemble methods** for optimal prediction accuracy

### Key Performance Metrics
- **Test AUC**: 0.8695
- **Test Accuracy**: 82.23%
- **Test F1-Score**: 0.8605
- **Test Precision**: 81.92%
- **Test Recall**: 90.62%
- **Optimal Threshold**: 0.470 (max_acc mode)

## 🏗️ System Architecture

The system follows a comprehensive pipeline with multiple specialized components:

```
Raw Data → Preprocessing → Model Training → Ensemble Fusion → Evaluation → UI Interface
```

### Core Components

1. **Data Preprocessing Pipeline** - Image segmentation, normalization, augmentation
2. **Machine Learning Models** - XGBoost, LightGBM for tabular data
3. **Deep Learning Models** - Multimodal CNN, MIL Attention for images
4. **Ensemble Fusion System** - Late fusion with meta-learning
5. **Interactive Web Interface** - Gradio-based UI for real-time inference

## 📁 Project Structure

```filetree
multimodal-tooth-restoration-ai/
├── src/
│   ├── config.py                          # Global configuration
│   ├── preprocessing/
│   │   ├── pipeline.py                    # Main preprocessing pipeline
│   │   ├── segment.py                     # Mask R-CNN segmentation
│   │   ├── normalise.py                   # CLAHE & geometric correction
│   │   ├── augment.py                     # Image augmentation utilities
│   │   ├── augment_records.py             # Augmentation with metadata
│   │   └── Standraized_dataset.py         # Tabular data preprocessing
│   ├── vision/
│   │   └── predict_vision.py              # Vision model inference
│   ├── tabular/
│   │   └── predict_tabular.py             # Tabular model inference
│   └── utils/
│       └── io.py                          # I/O utilities
├── experiments/
│   ├── multimodal_v1/
│   │   ├── train_mm_joint_dualtask.py     # Joint multimodal training
│   │   └── finalize_mm_dualtask_from_ckpts.py
│   ├── fusion_v1/
│   │   ├── stack_blend.py                 # Final ensemble fusion
│   │   └── stack_blend_optional.py
│   ├── vision_v2/
│   │   ├── train_hard_v2.py               # Progressive vision training
│   │   ├── ensemble_hard.py               # Vision ensemble
│   │   └── stacking/
│   └── data_v2/
│       └── make_balanced_splits.py        # Balanced data splitting
├── models/
│   ├── outputs/                           # Trained model artifacts
│   ├── segmenter/
│   │   └── mask_rcnn_molar.pt            # Tooth segmentation model
│   ├── vision/
│   │   ├── train_hard.py                 # Hard label vision training
│   │   └── train_soft.py                 # Soft label vision training
│   ├── xgboost_model.py                  # XGBoost training
│   ├── lightgbm_model.py                 # LightGBM training
│   └── stack_blend.py                    # Stacking ensemble
├── weights/
│   ├── mm_dualtask_v1/                   # Multimodal model weights
│   ├── mil_v1/                           # MIL attention weights
│   ├── fusion/                           # Fusion model weights
│   └── v2/                               # Enhanced vision weights
├── data/
│   ├── excel/
│   │   ├── data_processed.xlsx           # Processed tabular data
│   │   ├── data_dl.xlsx                  # Deep learning dataset
│   │   └── data_dl_augmented.xlsx        # Augmented dataset
│   ├── raw/images/                       # Original dental images
│   ├── processed/images/                 # Preprocessed images
│   └── augmented/                        # Augmented image dataset
├── ui/
│   └── gradio_app/
│       ├── app.py                        # Main Gradio interface
│       ├── infer_mm.py                   # Multimodal inference
│       ├── infer_mil.py                  # MIL inference
│       ├── tab_model.py                  # Tabular model interface
│       ├── stack_meta.py                 # Meta-learning stacker
│       └── utils.py                      # UI utilities
├── results/                              # Evaluation results
├── configs/
│   └── fusion.yaml                       # Fusion configuration
├── tests/
│   ├── evaluate_models.py                # Model evaluation
│   └── test_pipeline.py                  # Pipeline tests
├── run_pipeline.py                       # Preprocessing runner
├── run_fusion.py                         # Fusion model runner
├── run_augment_records.py                # Augmentation runner
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
torch >= 1.9.0
torchvision
sklearn
pandas
numpy
opencv-python
albumentations
gradio
xgboost
lightgbm
timm
joblib
Pillow
```

### Installation

```bash
git clone <repository-url>
cd multimodal-tooth-restoration-ai
pip install -r requirements.txt
```

### Complete Pipeline Execution

#### 1. Data Preprocessing

**Tabular Data Standardization:**
```bash
python src/preprocessing/Standraized_dataset.py
```

**Image Preprocessing with Multiple Options:**

```bash
# Default: Crop ON, Rotation ON (recommended)
python run_pipeline.py --input_dir data/raw/images --output_dir data/processed/images --model_path models/segmenter/mask_rcnn_molar.pt

# Crop OFF, Rotation ON
python run_pipeline.py --input_dir data/raw/images --output_dir data/processed_nocrop --model_path models/segmenter/mask_rcnn_molar.pt --no_crop

# Crop ON, Rotation OFF
python run_pipeline.py --input_dir data/raw/images --output_dir data/processed_norotate --model_path models/segmenter/mask_rcnn_molar.pt --no_rotate

# Crop OFF, Rotation OFF
python run_pipeline.py --input_dir data/raw/images --output_dir data/processed_nocrop_norotate --model_path models/segmenter/mask_rcnn_molar.pt --no_crop --no_rotate
```

#### 2. Data Augmentation

**Recommended Legacy Augmentation (10x expansion):**
```bash
python run_augment_records.py --input-table data/excel/data_dl.xlsx --images-src data/processed/images --images-dst data/augmented --num-aug-per-image 10 --make-val --val-frac 0.12 --aug-preset legacy --out-csv data/excel/data_dl_augmented.csv --out-xlsx data/excel/data_dl_augmented.xlsx
```

**Alternative Augmentation Presets:**

```bash
# Exact 10 methods with grouped validation
python run_augment_records.py --input-table data/excel/data_dl.xlsx --images-src data/processed/images --images-dst data/augmented --num-aug-per-image 10 --make-val --val-frac 0.12 --aug-preset ten --out-csv data/excel/data_dl_augmented.csv --out-xlsx data/excel/data_dl_augmented.xlsx

# Lighter PIL-only transforms
python run_augment_records.py --input-table data/excel/data_dl.xlsx --images-src data/processed/images --images-dst data/augmented --num-aug-per-image 10 --make-val --val-frac 0.12 --aug-preset simple --out-csv data/excel/data_dl_augmented.csv --out-xlsx data/excel/data_dl_augmented.xlsx

# No augmentation (copy originals only)
python run_augment_records.py --input-table data/excel/data_dl.xlsx --images-src data/processed/images --images-dst data/augmented --num-aug-per-image 0 --make-val --val-frac 0.12 --aug-preset none --out-csv data/excel/data_dl_augmented.csv --out-xlsx data/excel/data_dl_augmented.xlsx
```

#### 3. Train Individual Models

**Tabular Models:**

```bash
# XGBoost with optimized settings
python models/xgboost_model.py --tune-metric f1 --calibration sigmoid

# XGBoost without monotone constraints (alternative)
python models/xgboost_model.py --no-monotone

# LightGBM with calibration
python models/lightgbm_model.py

# LightGBM with softer weights (alternative)
python models/lightgbm_model.py --consensus-power 0.5 --no-monotone --calibration none
```

**Vision Models:**

```bash
# Hard labels (Direct vs Indirect) with grouped validation
python models/vision/train_hard.py --csv-path data/excel/data_dl_augmented.csv --images-root data/augmented --model-name tf_efficientnet_b3_ns --img-size 512 --epochs 20 --batch-size 12 --num-workers 4 --seed 42 --group-col origin_id --tta --tune-threshold

# Soft labels (probability of Indirect)
python models/vision/train_soft.py --csv-path data/excel/data_dl_augmented.csv --images-root data/augmented --model-name tf_efficientnet_b3_ns --img-size 512 --epochs 20 --batch-size 12 --num-workers 4 --seed 42 --group-col origin_id --tta
```

**Enhanced Vision Training (Progressive B4):**

```bash
# Progressive training with multiple seeds
python experiments/vision_v2/train_hard_v2.py --csv-path data/excel/data_dl_augmented.csv --images-root data/augmented --seeds 42,1337,2025 --save-dir weights/v2 --run-name hard_b4_prog

# Ensemble evaluation with TTA
python experiments/vision_v2/ensemble_hard.py --csv-path data/excel/data_dl_augmented.csv --images-root data/augmented --ckpts "weights/v2/hard_b4_prog_seed*_stage2_512.pt" --tta
```

**Multimodal Joint Training:**

```bash
python experiments/multimodal_v1/train_mm_joint_dualtask.py --xlsx data/excel/data_dl_augmented.xlsx --image-root data/augmented --outdir weights/mm_dualtask_v1 --backbone tf_efficientnet_b4_ns --epochs 30 --batch-size 32 --lr 3e-4 --alpha 1.0 --beta 0.3 --folds 5 --amp
```

#### 4. Model Evaluation

**Individual Model Evaluation:**

```bash
# XGBoost evaluation
python tests/evaluate_models.py --model xgb

# LightGBM evaluation
python tests/evaluate_models.py --model lgbm

# Both tabular models
python tests/evaluate_models.py --model both

# Vision models
python models/vision/eval_models.py --which both --hard-ckpt weights/vision_hard_best.pt --soft-ckpt weights/vision_soft_best.pt --csv-path data/excel/data_dl_augmented.csv --images-root data/augmented --batch-size 32 --out-csv outputs/both_preds.csv
```

**Ensemble Evaluation:**

```bash
# Blend tabular models
python tests/evaluate_models.py --model blend --tune both --tune-metric f1

# Stack blend with optimized parameters
python models/stack_blend.py --tune-metric accuracy --min-weight 0.15 --consensus-power 0.6
```

#### 5. Final Ensemble Training & Evaluation

**The Ultimate Fusion Command:**

```bash
python experiments/fusion_v1/stack_blend.py --xlsx_tab data/excel/data_processed.xlsx --oof_mm weights/mm_dualtask_v1/finalized/oof_val.csv --pred_mm weights/mm_dualtask_v1/finalized/pred_test.csv --oof_mil weights/mil_v1/oof_val.csv --pred_mil weights/mil_v1/pred_test.csv --outdir results/stack_v2 --thr-mode max_acc
```

**Alternative Fusion Approaches:**

```bash
# Using run_fusion.py (simplified interface)
python run_fusion.py train --calibrator isotonic --metric f1

# With custom tabular models
python run_fusion.py train --xgb-model models/outputs/xgb_classifier_pipeline.joblib --lgbm-model models/outputs/lgbm_regressor_pipeline.joblib

# Inspect fusion results
python run_fusion.py info
```

## 📊 Data Pipeline Details

### 1. Tabular Data Processing

**Input Dataset Structure:**
```
422 dental cases with clinical features:
- Numerical: depth, width (cavity dimensions)
- Binary: enamel_cracks, occlusal_load, carious_lesion, cervical_lesion
- Categorical: opposing_type, adjacent_teeth, age_range
- Labels: Direct/Indirect votes → p_indirect, y_majority
```

**Processing Steps:**
1. **Standardization**: Categorical encoding, missing value handling
2. **Feature Engineering**: Domain-specific transformations
3. **Label Generation**: 
   - `p_indirect`: Soft probability from expert votes
   - `y_majority`: Hard binary label (threshold = 0.5)
   - `weight`: Consensus strength for sample weighting
4. **Train/Test Split**: 80 test samples, remaining for training

### 2. Image Data Processing

**10-Step Augmentation Pipeline:**
1. **Horizontal flip** (probability-based)
2. **Vertical flip** (low probability)
3. **Translation** (shift via Affine transformation)
4. **Scaling** (zoom in/out via Affine)
5. **Rotation** (±25° via Affine)
6. **Brightness & contrast** adjustment
7. **Hue/saturation/value** color jitter
8. **Gaussian noise** (light application)
9. **Motion blur** (optional, can be disabled)
10. **Elastic deformation** (mild shape warping)

**Preprocessing Pipeline:**
```
Raw Image → Mask R-CNN Segmentation → CLAHE Enhancement → 
Geometric Correction → Augmentation (10x) → 512×512 Resize
```

### 3. Advanced Data Splitting

**Balanced Split Generation:**
```bash
python experiments/data_v2/make_balanced_splits.py --raw-xlsx data/excel/data.xlsx --processed-xlsx data/excel/data_processed.xlsx --dl-xlsx data/excel/data_dl.xlsx --dl-aug-xlsx data/excel/data_dl_augmented.xlsx --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 --seed 42 --group-col origin_id --label-col y_majority
```

## 🤖 Model Architecture Details

### 1. Tabular Models

**XGBoost Configuration:**
- **Objective**: Binary logistic regression
- **Monotonicity Constraints**: Applied to depth, width features
- **Class Balancing**: Automatic weight adjustment
- **Calibration**: Platt scaling for probability refinement
- **Hyperparameter Tuning**: Grid search with cross-validation

**LightGBM Configuration:**
- **Objective**: Regression on soft targets (p_indirect)
- **Categorical Features**: Native handling
- **Early Stopping**: Validation-based
- **Consensus Weighting**: Sample importance from expert agreement

### 2. Deep Learning Models

**Multimodal Joint Architecture (mm_dualtask_v1):**
```python
class MMJointModel(nn.Module):
    def __init__(self):
        # Image encoder
        self.backbone = timm.create_model('tf_efficientnet_b4_ns', 
                                        pretrained=True, 
                                        num_classes=0)  # Feature extractor
        
        # Tabular encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1792 + 64, 512),  # EfficientNet-B4 + tabular
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dual heads
        self.cls_head = nn.Linear(512, 1)  # Hard classification
        self.reg_head = nn.Linear(512, 1)  # Soft regression
```

**Loss Function:**
```python
total_loss = α × BCEWithLogitsLoss(hard_labels) + β × BCEWithLogitsLoss(soft_labels)
# α = 1.0, β = 0.3 for balanced learning
```

**MIL Attention Architecture:**
```python
class MILNet(nn.Module):
    def __init__(self):
        self.encoder = timm.create_model('tf_efficientnet_b0_ns')
        self.attention = GatedAttention(in_dim=1280, hidden_dim=256)
        self.classifier = nn.Linear(1280, 1)
```

### 3. Ensemble Fusion System

**Late Fusion Strategy:**
```python
# Meta-learner on out-of-fold predictions
meta_features = concat([pred_multimodal, pred_mil, pred_tabular])
meta_model = LogisticRegression()
final_prediction = meta_model.predict_proba(meta_features)
```

**Calibration Methods:**
- **Isotonic Regression**: Non-parametric calibration
- **Platt Scaling**: Sigmoid-based calibration
- **Temperature Scaling**: Learned temperature parameter

## 🎯 Training Strategy Details

### Cross-Validation Methodology

**GroupKFold Strategy:**
```python
# Ensures no data leakage by patient/origin grouping
cv = GroupKFold(n_splits=5)
for train_idx, val_idx in cv.split(X, y, groups=origin_id):
    # Train fold models
    # Generate out-of-fold predictions for meta-learning
```

### Advanced Training Techniques

**Progressive Training (Vision V2):**
```bash
# Stage 1: 384px training for 15 epochs
# Stage 2: 512px fine-tuning for 10 epochs
# Multiple seeds for ensemble diversity
```

**Test-Time Augmentation:**
```python
# Horizontal/vertical flips during inference
# Prediction averaging across augmentations
```

**Threshold Optimization:**
```python
def optimize_threshold(y_true, y_prob, metric='f1'):
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [metric_fn(y_true, y_prob >= t) for t in thresholds]
    return thresholds[np.argmax(scores)]
```

## 📈 Comprehensive Evaluation Results

### Final Performance Metrics

**Test Set Performance (Stack Blend v2):**
```
Threshold mode: max_acc | target: 0.8 | chosen thr: 0.470
=== OOF === {'auc': 0.8935, 'acc': 0.8456, 'prec': 0.8644, 'rec': 0.9053, 'f1': 0.8844}
=== TEST === {'auc': 0.8695, 'acc': 0.8223, 'prec': 0.8192, 'rec': 0.9062, 'f1': 0.8605}
```

### Individual Model Performance

| Model | Test AUC | Test Acc | Test F1 | Notes |
|-------|----------|----------|---------|--------|
| XGBoost | ~0.82 | ~0.78 | ~0.80 | Monotone constraints |
| LightGBM | ~0.84 | ~0.79 | ~0.82 | Soft target training |
| Vision Hard | ~0.81 | ~0.76 | ~0.79 | EfficientNet-B3 |
| Vision Soft | ~0.83 | ~0.78 | ~0.81 | ConvNeXt-Tiny |
| Multimodal | ~0.85 | ~0.80 | ~0.83 | Joint training |
| **Final Ensemble** | **0.8695** | **0.8223** | **0.8605** | **Meta-learning** |

### Clinical Relevance

**High Recall Priority:**
> In clinical workflow, missing an "Indirect" restoration (false negative) is typically more costly than over-recommending indirect treatment. The system achieves 90.62% recall, minimizing missed indirect cases.

## 🔧 Advanced Configuration

### Hyperparameter Optimization

**Tabular Models:**
```bash
# XGBoost with custom parameters
python models/xgboost_model.py --tune-metric accuracy --min-weight 0.15 --consensus-power 0.6 --no-monotone

# LightGBM with consensus weighting
python models/lightgbm_model.py --consensus-power 0.6 --calibration isotonic
```

**Vision Models:**
```bash
# Progressive training with custom architecture
python experiments/vision_v2/train_hard_v2.py --model-name tf_efficientnet_b4_ns --img-size 512 --epochs 25 --lr 3e-4 --weight-decay 1e-5

# Multi-seed ensemble training
python experiments/vision_v2/train_hard_v2.py --seeds 42,1337,2025,777,999 --ensemble-size 5
```

**Fusion Configuration:**
```yaml
# configs/fusion.yaml
calibrator: isotonic        # isotonic | platt | temperature
threshold_metric: f1        # f1 | youden | pr_auc | accuracy
ensemble_method: stacking   # stacking | blending | voting
meta_model: logistic        # logistic | ridge | random_forest
```

### Inference Modes

**Single Case Inference:**
```bash
# By image name
python run_fusion.py infer-one --image-name 133.jpg

# By test set index
python run_fusion.py infer-one --row-idx 5

# With custom threshold
python run_fusion.py infer-one --threshold 0.3 --out results/case_analysis.json
```

**Batch Inference:**
```bash
# Test set only
python run_fusion.py infer-batch --split test

# All data
python run_fusion.py infer-batch --split all --out results/all_predictions.csv

# External dataset
python run_fusion.py infer-batch --csv-in external_data.csv --out results/external_predictions.csv
```

## 🚨 Important Considerations

### Data Requirements

**Image Quality:**
- **Minimum Resolution**: 512×512 pixels recommended
- **Format Support**: JPG, PNG, TIFF, BMP
- **Quality**: Clear dental radiographs or clinical photos
- **Orientation**: Automatic rotation correction available

**Clinical Data Completeness:**
- **Required Features**: All 9 clinical parameters
- **Missing Data**: Automatic imputation available but may reduce accuracy
- **Data Ranges**: 
  - `depth`, `width`: Continuous values (0-2+ typical range)
  - Binary features: {0, 1}
  - Categorical: Properly encoded values

### Hardware Requirements

**Training Requirements:**
- **GPU**: 8GB+ VRAM (RTX 3070/V100 or better)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ for full dataset with augmentation
- **CPU**: Multi-core for data loading (8+ cores recommended)

**Inference Requirements:**
- **GPU**: 4GB+ VRAM (optional, CPU inference supported)
- **RAM**: 8GB+ system memory
- **Storage**: 5GB+ for model weights

### Performance Optimization

**Memory Management:**
```python
# Gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Mixed precision training
scaler = GradScaler()
with autocast():
    loss = model(images, features)
```

**Batch Size Guidelines:**
- **Training**: 16-32 (adjust based on GPU memory)
- **Inference**: 64-128 (higher for batch processing)
- **Augmentation**: Process in chunks to avoid memory overflow

### Known Limitations

**Model Generalization:**
- Trained on specific dental imaging protocols
- Performance may vary with different imaging systems
- Requires similar patient demographics for optimal results

**Clinical Validation:**
- **Research Use Only**: Requires clinical validation for medical use
- **Expert Review**: Predictions should be reviewed by dental professionals
- **Regulatory Approval**: Not approved for clinical diagnosis

**Technical Constraints:**
- **Internet Connection**: Required for downloading pre-trained weights
- **File Formats**: Limited to supported image/data formats
- **Processing Time**: Scales with batch size and augmentation settings

## 🔍 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# CUDA compatibility issues
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Missing dependencies
pip install -r requirements.txt --upgrade

# Albumentations installation
pip install albumentations --no-binary albumentations
```

**Memory Issues:**
```bash
# Reduce batch size
--batch-size 8

# Disable augmentation during debugging
--num-aug-per-image 0

# Use CPU inference
--device cpu
```

**Data Loading Errors:**
```bash
# Check file paths
python -c "from pathlib import Path; print(Path('data/excel/data_processed.csv').exists())"

# Verify image directory
ls data/processed/images/*.jpg | head -5

# Validate CSV format
python -c "import pandas as pd; print(pd.read_csv('data/excel/data_processed.csv').head())"
```

### Performance Debugging

**Model Loading Issues:**
```python
# Check checkpoint compatibility
import torch
ckpt = torch.load('weights/model.pt', map_location='cpu')
print(list(ckpt.keys()))
```

**Inference Speed Optimization:**
```python
# Enable JIT compilation
model = torch.jit.script(model)

# Use half precision
model.half()
```

## 🖥️ User Interface - Web Application for Clinical Use

### 🚀 Quick Launch for End Users

The system includes a **user-friendly web interface** designed for dental professionals and researchers who want to use the trained models without technical expertise.

#### Launch the Interface

```bash
# Navigate to project directory
cd multimodal-tooth-restoration-ai

# Launch the web application
python ui/gradio_app/app.py

# Access the interface at: http://localhost:7860
```

The interface will automatically open in your default web browser. If not, manually navigate to `http://localhost:7860`.

### 🎯 Interface Features

#### **Main Prediction Panel**

**Image Upload & Processing:**
- **Drag & Drop**: Upload dental radiographs or clinical photos
- **Format Support**: JPG, PNG, TIFF, BMP formats accepted
- **Resolution Check**: Automatic validation for minimum 512×512 pixels
- **Real-time Preview**: See uploaded image before processing

**Preprocessing Options:**
- **Automatic Segmentation**: Toggle tooth detection and cropping
- **Rotation Correction**: Enable/disable automatic image orientation
- **Custom Model Path**: Specify alternative segmentation models

**Clinical Data Entry (Optional):**
- **Numerical Fields**: 
  - `depth` (mm): Cavity depth measurement
  - `width` (mm): Cavity width measurement
- **Categorical Selections**:
  - `enamel_cracks`: Presence of enamel damage
  - `occlusal_load`: Chewing force considerations
  - `carious_lesion`: Decay indicators
  - `opposing_type`: Type of opposing tooth
  - `adjacent_teeth`: Adjacent tooth status
  - `age_range`: Patient age category
  - `cervical_lesion`: Gum line involvement

**Prediction Configuration:**
- **Threshold Mode**: Choose optimization strategy
  - `max_acc`: Maximum accuracy (default)
  - `max_f1`: Maximum F1-score
  - `youden`: Youden's J statistic
  - `target_prec`: Target precision
  - `target_rec`: Target recall

#### **Results Display**

**Primary Prediction:**
- **Classification**: **Direct** or **Indirect** restoration recommendation
- **Confidence Score**: Calibrated probability with threshold information
- **Threshold Details**: Selected threshold and optimization mode

**Model Contributions:**
- **Per-Stream Analysis**: Individual model predictions
  - Tabular Model (LightGBM/XGBoost)
  - Multimodal Model (Image + Clinical)
  - MIL Model (Image-only attention)
- **Ensemble Details**: How models are combined

**Visual Feedback:**
- **Processed Image**: Shows preprocessed tooth image
- **Quality Indicators**: Image processing success/failure status

#### **Performance Dashboard**

**System Metrics:**
- **Current Model Performance**: Live loading of latest evaluation results
- **Detailed Breakdown**: AUC, Accuracy, Precision, Recall, F1-Score
- **Validation vs Test**: Out-of-fold and test set performance

### 📋 Usage Workflows

#### **Workflow 1: Image-Only Classification**

1. **Upload Image**: Drag dental radiograph to upload area
2. **Configure Processing**: 
   - Keep default settings (crop ON, rotate ON)
   - Select threshold mode (max_acc recommended)
3. **Run Prediction**: Click "Preprocess & Predict"
4. **Review Results**: 
   - See Direct/Indirect recommendation
   - Check confidence score
   - Examine processed image quality

#### **Workflow 2: Full Multimodal Classification** (Recommended)

1. **Upload Image**: Provide dental radiograph
2. **Enter Clinical Data**: Fill ALL clinical fields
   - Measure depth and width
   - Select appropriate categorical values
   - Ensure no fields are left empty
3. **Run Prediction**: Use "Preprocess & Predict"
4. **Analyze Results**:
   - Compare individual model predictions
   - Note how clinical data influences final decision
   - Review ensemble contribution weights

#### **Workflow 3: Clinical Data Only**

1. **Skip Image Upload**: Leave image field empty
2. **Complete Clinical Form**: Provide all 9 clinical parameters
3. **Run Prediction**: System will use tabular model only
4. **Interpret Results**: Understand limitation of image-free prediction

### 🏥 Clinical Integration Guidelines

#### **For Dental Practices**

**Setup Requirements:**
- **Hardware**: Standard laptop/desktop (GPU optional)
- **Internet**: Required for initial model download only
- **Training**: 15-minute orientation for dental staff

**Integration Steps:**
1. **Install System**: Follow installation instructions
2. **Launch Interface**: Run `python ui/gradio_app/app.py`
3. **Staff Training**: Train personnel on image capture and data entry
4. **Workflow Integration**: Incorporate into existing patient examination process

**Best Practices:**
- **Image Quality**: Ensure clear, well-lit radiographs
- **Consistent Measurements**: Use standardized measurement protocols
- **Expert Review**: Always have licensed dentist review AI recommendations
- **Documentation**: Record AI predictions alongside clinical notes

#### **For Research Institutions**

**Batch Processing:**
- **Multiple Cases**: Process entire patient databases
- **Comparative Studies**: Compare AI vs. expert decisions
- **Performance Tracking**: Monitor accuracy across different populations

**Data Export:**
- **CSV Output**: Export predictions for statistical analysis
- **Integration**: Connect with electronic health records
- **Audit Trail**: Maintain logs of all predictions

### 🛡️ Safety & Compliance

#### **Clinical Disclaimers**

**⚠️ IMPORTANT MEDICAL DISCLAIMERS:**

1. **Research Use Only**: This system is designed for research and educational purposes
2. **Not a Medical Device**: Not approved by FDA or other regulatory bodies
3. **Expert Review Required**: All predictions must be reviewed by licensed dental professionals
4. **Clinical Responsibility**: Final treatment decisions remain with the dentist
5. **Performance Limitations**: Accuracy may vary with different patient populations

#### **Data Privacy**

**Local Processing:**
- **No Cloud Upload**: All processing happens on local machine
- **Data Retention**: Temporary files are cleaned up automatically
- **HIPAA Considerations**: Consult with compliance officer for patient data handling

**Session Management:**
- **Temporary Storage**: Images stored in `/ui/tmp/` during processing
- **Automatic Cleanup**: Session data removed after processing
- **No Logging**: Patient data not logged unless explicitly enabled

### 🎓 Training & Support

#### **User Training Materials**

**Quick Start Guide:**
1. **System Overview**: Understanding the multimodal approach
2. **Image Guidelines**: Best practices for dental photography
3. **Clinical Data Entry**: Standardized measurement protocols
4. **Result Interpretation**: Understanding confidence scores and thresholds

**Video Tutorials:** (To be created)
- Interface walkthrough
- Clinical workflow integration
- Troubleshooting common issues

#### **Advanced Features**

**Threshold Customization:**
- **Clinical Priorities**: Adjust based on practice philosophy
- **Risk Tolerance**: Balance sensitivity vs. specificity
- **Population-Specific**: Optimize for patient demographics

**Model Selection:**
- **Ensemble vs. Individual**: Compare different model approaches
- **Confidence Thresholding**: Set minimum confidence requirements
- **Fallback Strategies**: Handle low-confidence predictions

### 📊 Interface Technical Specifications

**Performance:**
- **Response Time**: 2-10 seconds per prediction (depending on hardware)
- **Concurrent Users**: Single-user interface (expandable)
- **Resource Usage**: 2-4GB RAM, optional GPU acceleration

**Browser Compatibility:**
- **Chrome**: Recommended (version 90+)
- **Firefox**: Supported (version 88+)
- **Safari**: Supported (version 14+)
- **Edge**: Supported (version 90+)

**Accessibility:**
- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: Compatible with ARIA standards
- **Mobile Responsive**: Works on tablets (1024px+ width recommended)

### 🔧 Customization Options

**Interface Themes:**
```python
# Modify ui/gradio_app/app.py
theme = gr.themes.Soft(
    primary_hue="blue",      # Change to "green", "red", etc.
    secondary_hue="blue"
)
```

**Default Settings:**
```python
# Modify DEFAULTS dictionary in app.py
DEFAULTS = {
    "thr_mode": "max_f1",    # Change default threshold mode
    "thr_target": 0.85,      # Adjust target threshold
    # ... other settings
}
```

**Clinical Field Customization:**
- **Add New Fields**: Extend tabular feature list
- **Modify Choices**: Update dropdown options
- **Validation Rules**: Implement custom data validation

---

**For technical support with the UI system, please contact the development team or refer to the troubleshooting section above.**

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd multimodal-tooth-restoration-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

### Code Style

```bash
# Format code
black src/ experiments/ models/

# Lint code
flake8 src/ experiments/ models/

# Type checking
mypy src/
```

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Test specific functionality
python tests/test_pipeline.py

# Integration tests
python tests/evaluate_models.py --model both
```

### Adding New Features

1. **Fork Repository**
2. **Create Feature Branch**: `git checkout -b feature/new-enhancement`
3. **Implement Changes**: Follow existing code patterns
4. **Add Tests**: Ensure proper test coverage
5. **Update Documentation**: Modify README and docstrings
6. **Submit Pull Request**: Include detailed description

## 📄 License & Citation

### License
This project is licensed under Private License. See LICENSE file for details.

### Citation

If you use this system in your research, please cite:

```bibtex
@software{multimodal_tooth_restoration_2024,
  title={Multimodal Classification System for Tooth Restoration Types Using Hybrid Machine Learning Techniques},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]},
  note={Hybrid ensemble system combining clinical features and dental imaging for restoration type classification}
}
```

### Academic References

```bibtex
@article{tooth_restoration_classification_2024,
  title={Automated Classification of Dental Restoration Types Using Multimodal Deep Learning},
  journal={Journal of Dental Informatics},
  year={2024},
  note={Submitted for review}
}
```

## 📞 Support & Contact

### Getting Help

**Professional Support:**
- **Email**: [ahmed1992madrid@gmail.com]
- **LinkedIn**: [https://www.linkedin.com/in/ahmed-majid-918866106/]
- **ResearchGate**: [https://www.researchgate.net/profile/Ahmed-Majid-3]

### Reporting Issues

When reporting bugs, please include:
1. **System Information**: OS, Python version, GPU details
2. **Error Logs**: Full stack trace
3. **Reproduction Steps**: Minimal example to reproduce the issue
4. **Data Context**: Dataset size, image resolution, etc.

### Feature Requests

For new features, please provide:
1. **Use Case**: Detailed description of the need
2. **Expected Behavior**: How the feature should work
3. **Implementation Ideas**: Suggested approach (optional)
4. **Priority Level**: Critical/High/Medium/Low

---

**Acknowledgments:** This project builds upon numerous open-source libraries and research contributions from the computer vision and machine learning communities. Special thanks to the timm library authors, PyTorch team, and the broader scientific Python ecosystem.

**Disclaimer:** This system is designed for research and educational purposes. Clinical applications require appropriate validation, regulatory approval, and expert medical oversight. The authors assume no responsibility for clinical decisions made based on this system's outputs.