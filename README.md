HSDRNet
HSDRNet: A Hard-Aware Semantic Detail Recalibration Network for Remote Sensing Change Detection

Yao Lin and Libo He
Computer Vision Laboratory, Chongqing University of Technology, Chongqing, China
Overview
HSDRNet detects changes between registered bi-temporal remote sensing images through a calibration-before-fusion strategy. It connects difficulty-guided local modelling with semantic control of detail injection.

The network uses a shared ResNet18d-based encoder with pyramid context aggregation and progressive feature fusion. Absolute temporal differences are computed at four feature levels. A hard-region estimation branch provides a learned difficulty representation, while cross-layer review refines the high-level difference features. The HSDR module consists of:
- Hard-aware Detail Calibration (HDC): modulates low-level temporal differences using the hard-region feature before extracting local structures with parallel 3 × 3, 5 × 5, and 7 × 7 convolutional branches.
- Semantic Consensus Modelling (SCM): adaptively aggregates reviewed high-level differences using channel weights normalised across feature levels.
- Semantic-gated Detail Injection (SGDI): jointly processes the semantic consensus and calibrated details to generate a spatial gate for detail injection.
- Residual Prediction Refinement (RPR): refines the fused representation before final change prediction.
Source Files
File	Description
models/encoder.py	Shared ResNet18d encoder, pyramid pooling, and feature fusion
models/decoder.py	Prediction heads, cross-layer review, and hard-region estimation
models/hsdr.py	HDC, SCM, SGDI, and RPR
models/hsdrnet.py	Complete model and forward interface
models/model.py	Model entry point
train.py	Training, validation, checkpoint selection, and final evaluation
test.py	Checkpoint evaluation and prediction visualisation
tools/sanity_check.py	Output-shape checks and optional checkpoint compatibility check
utils/checkpoint.py	Checkpoint extraction and strict loading utilities
utils/metric_tool.py	Confusion-matrix-based evaluation metrics


Environment and Required Files
Install PyTorch and torchvision for your CUDA environment using the official installation instructions. Then install the remaining directly imported dependencies:
python -m pip install timm numpy scipy Pillow
The supplied archive does not specify the original Python, PyTorch, torchvision, or timm versions. Use mutually compatible versions; an experimentally verified version lock is not provided. The data-loading modules may require additional packages.

Run all commands below from the project root containing train.py and test.py.
Data-loading modules
Both entry scripts require the following project files, which must be restored from the original experiment project before training or evaluation:
- datasets/dataset.py
- datasets/Transforms.py
These files are absent from the supplied archive. dataset.py must expose the Dataset class used by the entry scripts. Transforms.py must provide Compose, Normalize, Scale, RandomCropResize, RandomFlip, RandomExchange, and ToTensor.

Each dataset sample must return an image tensor of shape (6, H, W) and a binary label tensor of shape (1, H, W). The first three image channels represent the first acquisition, and the remaining three represent the second. Labels must use 0 for unchanged pixels and 1 for changed pixels. The evaluation script also uses the dataset's file_list attribute to name visualisations.
Encoder weights
The current encoder loads ImageNet-pretrained ResNet18d weights from this exact relative path:
models/resnet18d_ra2-48a79e06.pth
Place the matching pretrained weight file at this location before constructing the model. It is not included in the supplied archive. This requirement also applies to test.py and tools/sanity_check.py, because both construct the encoder before loading any complete HSDRNet checkpoint.
Datasets
Download datasets: LEVIR, BCDD.

The manuscript uses LEVIR+ (LEVIR-CD+). The linked LEVIR repository provides both LEVIR-CD+ and S2Looking; select LEVIR-CD+ for the experiments described here. BCDD refers to the building change detection data available from the linked WHU dataset page.

The manuscript uses non-overlapping 512 × 512 patches for LEVIR+ and 512 × 512 test patches for BCDD:
Dataset	Training pairs	Validation pairs	Test pairs	Role
LEVIR+	1,780	768	1,392	Training and in-domain evaluation
BCDD	—	—	1,827	Direct out-of-domain evaluation


The default dataset roots are:
./datasets/BCD/LEVIR+512
./datasets/BCD/BCDD-512
The entry scripts request the train, val, and test splits for LEVIR+, and the test split for BCDD. The exact image subdirectories, filename pairing, and split-file conventions are defined by datasets/dataset.py; restore that loader before arranging the data. Cropping scripts and experiment-specific split lists are not included in the supplied archive. Dataset downloads alone do not establish the exact experimental split.

The entry scripts configure ImageNet-style normalisation with the following values, repeated for both temporal images:
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
These values follow BGR channel order. Preserve the original loader's channel convention and verify that Normalize applies the expected intensity scaling. Apply spatial augmentation consistently to both images and the label.
Training
Train on LEVIR+ and evaluate the selected checkpoint on LEVIR+ and BCDD:
python train.py \
  --model_name HSDRNet \
  --file_root LEVIR \
  --exp_tag levirplus \
  --data_root ./datasets/BCD/LEVIR+512 \
  --test_data_root ./datasets/BCD/LEVIR+512 \
  --test_data_root_2 ./datasets/BCD/BCDD-512 \
  --test_name LEVIR_Test \
  --test_name_2 BCDD_Test \
  --in_width 512 \
  --in_height 512 \
  --batch_size 16 \
  --max_steps 5000 \
  --lr 0.0005 \
  --seed 1234 \
  --save_dir ./weights \
  --on_gpu true
To train and evaluate on LEVIR+ only, use the same command with --test_data_root_2 "".

The training script uses AdamW with weight decay 0.01 and betas (0.9, 0.99). Each of the five change predictions receives BCE plus Dice loss. The hard-region branch receives BCE against the detached final prediction error, and its loss is added with unit weight. Auxiliary change predictions are generated before cross-layer review.

The implementation uses polynomial learning-rate decay with power 0.9. Its linear warm-up condition is restricted to the first epoch and global iterations below 200. The nominal --max_steps budget is converted to a whole number of epochs; training does not stop partway through the final epoch. With 1,780 training samples, batch size 16, and drop_last=True, --max_steps 5000 gives 46 epochs and 5,106 optimizer updates. Training requires at least one complete batch; retain a batch size of at least two because the pyramid pooling module uses batch normalisation after 1 × 1 pooling.

The checkpoint with the highest validation F1-score is saved as best_model.pth. After training, the script reloads this checkpoint and evaluates the specified test sets. BCDD is used only for evaluation; no BCDD fine-tuning is performed.

For the command above, the checkpoint is saved to:
weights/HSDRNet/LEVIR_levirplus_iter_5000_lr_0.0005_seed_1234/best_model.pth
Training and validation scores are written to trainValLog.txt in the same directory.

The manuscript reports five seeds: 1234, 3407, 4079, 42, and 777. Repeat the command with each seed to obtain separate runs. The scripts do not aggregate the mean and standard deviation automatically. CUDA operations are not configured for strict determinism, so setting a seed does not guarantee identical results across environments.
Evaluation
Evaluate a checkpoint produced by the training command above:
python test.py \
  --model_name HSDRNet \
  --file_root LEVIR \
  --exp_tag levirplus \
  --data_root ./datasets/BCD/LEVIR+512 \
  --test_data_root ./datasets/BCD/LEVIR+512 \
  --test_data_root_2 ./datasets/BCD/BCDD-512 \
  --test_name LEVIR_Test \
  --test_name_2 BCDD_Test \
  --in_width 512 \
  --in_height 512 \
  --batch_size 1 \
  --max_steps 5000 \
  --lr 0.0005 \
  --seed 1234 \
  --save_dir ./weights \
  --on_gpu true
Keep --model_name, --file_root, --exp_tag, --max_steps, --lr, --seed, and --save_dir consistent with training, because these arguments determine the checkpoint path. If --exp_tag is omitted, the basename of --data_root also contributes to the experiment directory name. The unmodified training and evaluation scripts have different default seeds (43 and 48), so specify the seed explicitly.

Although both entry scripts define --weight, the current implementation does not use this argument to load a checkpoint. Evaluation loads best_model.pth from the generated experiment directory and expects a plain model state dictionary, as saved by train.py.

Evaluation thresholds the final change probability at 0.5 and reports Kappa, IoU, F1-score, recall, precision, and overall accuracy. Metric values stored by the scripts are fractions; multiply by 100 to express percentages.

The default visualisation directories for this example are:
predict/HSDRNet/LEVIR/LEVIR_Test_seed1234/
predict/HSDRNet/LEVIR/BCDD_Test_seed1234/
The saved images are error maps: white indicates true positives, black true negatives, red false positives, and blue false negatives. These maps require ground-truth labels. Aggregate metrics are saved as results.mat in each visualisation directory, and evaluation scores are appended to Test.txt in the checkpoint directory.
Model Interface and Optional Sanity Check
import torch
from models.model import get_model

model = get_model()
model.eval()

inputs = torch.randn(1, 6, 512, 512)
with torch.no_grad():
    change_predictions, hard_region_map = model(inputs)

# Expected shapes:
# change_predictions: (1, 5, 512, 512)
# hard_region_map:    (1, 1, 512, 512)
final_change_probability = change_predictions[:, 0:1]
The five prediction channels are ordered as final prediction, D2 prediction, D3 prediction, D4 prediction, and D5 prediction. Both returned tensors contain sigmoid probabilities.

To check the output structure:
python tools/sanity_check.py
To additionally check a complete HSDRNet checkpoint:
python tools/sanity_check.py \
  --checkpoint ./weights/HSDRNet/LEVIR_levirplus_iter_5000_lr_0.0005_seed_1234/best_model.pth
Unlike test.py, this utility also supports common checkpoint wrappers and optional module. prefixes through utils/checkpoint.py. It checks compatibility and output dimensions; it does not establish benchmark accuracy.
Results Reported in the Manuscript
The following results are reported as mean ± standard deviation over five runs. They are manuscript results, not independently reproduced measurements from this README preparation.
Evaluation setting	Precision (%)	Recall (%)	F1 (%)	IoU (%)	Kappa (%)
LEVIR+	87.22 ± 0.81	83.98 ± 0.79	85.56 ± 0.31	74.77 ± 0.48	84.96 ± 0.33
BCDD, direct transfer from LEVIR+	78.51 ± 1.25	77.04 ± 1.42	77.77 ± 1.20	63.64 ± 1.61	76.81 ± 1.24


Contact
For questions about the method or experimental materials, contact the corresponding author, Libo He, at libohe2026@126.com.
