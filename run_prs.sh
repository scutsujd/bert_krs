# （1）personas_self_original 使用BertForPersonaResponseSelection 用的是mean方式
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg.txt 2>&1 &

# （2）personas_self_original 使用BertForPersonaResponseSelection_DIM
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_DIM.txt 2>&1 &

# （3）personas_self_original 使用BertForPersonaResponseSelection_twologits
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_twologits.txt 2>&1 &

# （4）personas_self_original 继承于(1) 使用BertForPersonaResponseSelection 采用mean_max方式
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrain_meanmax.txt 2>&1 &

# （5）personas_self_original 继承于(1) 使用BertForPersonaResponseSelection 采用mean方式
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrain_mean.txt 2>&1 &

# （6）personas_self_original 继承于(5) 使用BertForPersonaResponseSelection 采用mean方式 使用预训练好的模型
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrain_mean_lr5e-5.txt 2>&1 &

# （7）personas_self_original
#  BertForPersonaResponseSelection_twoloss 3e-5 mean 采用预训练过的模型
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrain_twoloss.txt 2>&1 &

# （8）personas_self_original
# BertForPersonaResponseSelection_multi_task
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrain_multi_task.txt 2>&1 &

# （9）BertForPersonaResponseSelection_V2
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrain_v2.txt 2>&1 &

# (10) BertForPersonaResponseSelection_V2
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_multi_task_V2.txt 2>&1 &

# (11) BertForPersonaResponseSelection_V2
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V2.txt 2>&1 &

# (11) BertForPersonaResponseSelection_V3
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V3_lw.txt 2>&1 &

PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V3.5_lw.txt 2>&1 &

# (12) BertForPersonaResponseSelection_V3
# personachat_self_revised 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_revised_noneg_pretrainV2_V3_lw.txt 2>&1 &

# (13) BertForPersonaResponseSelection_V3
# cmudog数据集 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2_V3_lw.txt 2>&1 &

# (14) BertForPersonaResponseSelection_V3
# personachat_self_original 数据集
# 消融实验，不用经过进一步预训练模型，直接用原版的BERT
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_nopretrain_V3_lw.txt 2>&1 &

# (15) BertForPersonaResponseSelection_V3
# personachat_self_revised 数据集
# 消融实验，不用经过进一步预训练模型，直接用原版的BERT
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_revised_noneg_nopretrain_V3_lw.txt 2>&1 &

# (16) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，不用经过进一步预训练模型，直接用原版的BERT
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_nopretrain_V3_lw.txt 2>&1 &

# (17) BertForPersonaResponseSelection_V3
# personachat_self_original 数据集
# 消融实验，取消context和knowledge的交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V3nointeract_lw.txt 2>&1 &

# (18) BertForPersonaResponseSelection_V3
# personachat_self_revised 数据集
# 消融实验，取消context和knowledge的交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_revised_noneg_pretrainV2_V3nointeract_lw.txt 2>&1 &

# (19) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，取消context和knowledge的交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2_V3nointeract_lw.txt 2>&1 &

# (20) BertForPersonaResponseSelection_V3
# personachat_self_original 数据集
# 消融实验，response 只和 context 交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V3nopersona_lw.txt 2>&1 &

# (21) BertForPersonaResponseSelection_V3
# personachat_self_revised 数据集
# 消融实验，response 只和 context 交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_revised_noneg_pretrainV2_V3nopersona_lw.txt 2>&1 &

# (22) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，response 只和 context 交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2_V3nopersona_lw.txt 2>&1 &

# (23) BertForPersonaResponseSelection_V3
# personachat_self_original 数据集
# 消融实验，response 只和 persona 交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V3nocontext_lw.txt 2>&1 &

# (24) BertForPersonaResponseSelection_V3
# personachat_self_revised 数据集
# 消融实验，response 只和 persona 交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_revised_noneg_pretrainV2_V3nocontext_lw.txt 2>&1 &

# (25) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，response 只和 persona 交互
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2_V3nocontext_lw.txt 2>&1 &

# (26) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，预训练模型只用MLM训练 没有CD
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2noCD_V3_lw.txt 2>&1 &

# (27) BertForPersonaResponseSelection_V3
# cmudog 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2noMLM_V3_lw.txt 2>&1 &

PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2noMLM_V3_2_lw.txt 2>&1 &

# (28) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，预训练模型取消数据增强(DA)
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainnoDA_V3_lw.txt 2>&1 &

# (29) BertForPersonaResponseSelection_V3
# personachat_self_original 数据集
# 消融实验，取消matching
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_original_noneg_pretrainV2_V3nomatching_lw.txt 2>&1 &

# (30) BertForPersonaResponseSelection_V3
# personachat_self_revised 数据集
# 消融实验，取消matching
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_personachat_self_revised_noneg_pretrainV2_V3nomatching_lw.txt 2>&1 &

# (31) BertForPersonaResponseSelection_V3
# cmudog 数据集
# 消融实验，取消matching
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_train.py > log_bert_cmudog_noneg_pretrainV2_V3nomatching_lw.txt 2>&1 &

# (32) Polyencoder_v1
# personachat_self_original 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_original_polyencoder_lw.txt 2>&1 &

# (32) Polyencoder_v1
# personachat_self_original 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_original_polyencoder_lr5e05_360codes_lw.txt 2>&1 &

# (33) Polyencoder_v1
# personachat_self_original 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_original_polyencoder_lr3e05_360codes_lw.txt 2>&1 &

# (34) Polyencoder_v1
# personachat_self_revised 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_revised_polyencoder_lw.txt 2>&1 &
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_revised_polyencoder_lr3e05_360codes_lw.txt 2>&1 &

# (35) Polyencoder_v1
# cmudog 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_cmudog_polyencoder_lr3e05_360codes_lw.txt 2>&1 &
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_cmudog_polyencoder_lr3e05_64codes_lw.txt 2>&1 &
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_cmudog_polyencoder_lr3e05_360codes_cls_lw.txt 2>&1 &

# (36) Biencoder_v1
# personachat_self_original 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_original_biencoder_lw.txt 2>&1 &

# (37) Biencoder_v1
# personachat_self_revised 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_personachat_self_revised_biencoder_lw.txt 2>&1 &

# (38) Biencoder_v1
# cmudog 数据集
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_cmudog_biencoder_lw.txt 2>&1 &
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/polyencoder/polyencoder_train.py > log_bert_cmudog_biencoder_2_lw.txt 2>&1 &


PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_create_pretraining_data.py


# ==============================================================================================
# 预训练
# (1) 保存在/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_bert_personachat_self_original/
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_personachat_self_original.txt 2>&1 &

# (2) 保存在/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_bert_personachat_self_original_V2
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_personachat_self_original_V2.txt 2>&1 &

# (3) 保存在/data/mgw_data/BERT_persona_response_selection/save_model/pretrain_bert_personachat_self_revised_V2
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_personachat_self_revised_V2.txt 2>&1 &

# (4) cmudog数据
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_cmudog_V2.txt 2>&1 &

# (5) cmudog数据
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_cmudog_V2.txt 2>&1 &

# (6) cmudog数据
# 消融实验，训练目标只保留MLM
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_cmudog_V2noCD.txt 2>&1 &

# (7) cmudog数据
# 消融实验，训练目标只保留CD
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_cmudog_V2noMLM.txt 2>&1 &

# (8) cmudog数据
# 消融实验，cmudog预训练数据减少，即没有数据增强data agumentation
PYTHONPATH=/home/mgw/persona_FIRE:$PYTHONPATH /home/mgw/miniconda3/envs/mgw_tf_torch/bin/python -u /home/mgw/persona_FIRE/LM_paper/Persona_Response_Selection_noneg/prs_pretraining.py > log_pretraining_bert_cmudog_noDA.txt 2>&1 &

