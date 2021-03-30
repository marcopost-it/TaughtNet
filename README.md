# TaughtNet

This repository contains the code for *TaughtNet: Knowledge Distillation from Deep Bidirectional Transformers for Multi-task Biomedical Named Entity Recognition*. The paper introduces *TaughtNet*, a knowledge distillation based framework which allows to leverage multiple data sources for Named Entity Recognition. For each data source, we train a teacher which can be considered as an "expert" in recognizing a specific entity type (the entity type for which the dataset has been annotated for). Then, a *student* distills the knowledge of multiple teachers to learn to recognize multiple entity types.

## 🔧 Setup
All requirements for *TaughtNet* can be found in <code>requirements.txt</code>. You can install all required packages with <code>pip install -r requirements.txt</code>

## 💻 Usage

To train your own taughtNet, you need to follow these steps:

1. **Train teachers**

        !python train_teacher.py \
        --data_dir 'data/BC2GM' \
        --model_name_or_path 'dmis-lab/biobert-base-cased-v1.1' \
        --output_dir 'models/Teachers/BC2GM' \
        --logging_dir 'models/Teachers/BC2GM' \
        --save_steps 10000 
  
2. **Aggregate datasets**: generate *global* datasets for teachers (to get their predictions over all the set of data) and student (to train it)

        !python generate_global_datasets.py

3. **Generate teachers distribution**: aggregate the output distributions of teachers to get the general distribution for all the entity types

        !python generate_teachers_distributions.py \
        --data_dir 'data' \
        --teachers_dir 'models/Teachers' \
        --model_name_or_path 'dmis-lab/biobert-base-cased-v1.1'

4. **Train the student!**

        !python train_student.py \
        --data_dir 'data/GLOBAL/Student' \
        --model_name_or_path 'dmis-lab/biobert-base-cased-v1.1' \
        --output_dir 'models/Student' \
        --logging_dir 'models/Student' \
        --save_steps 596 

The notebook <code>example.ipynlb</code> reported in this repo contains an example by using the following datasets: *BC2GM*, *BC5CDR-chem*, *NCBI-disease*

## 📕 Citation
Not yet published 
