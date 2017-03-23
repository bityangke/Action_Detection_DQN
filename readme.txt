th Hjj_Training.lua -gpu [] -data_path ActivityNet -class [] -alpha [] -log_err ./log/training_error_[].log -log_log ./log/log[] -batch_size [] -replay_buffer [] -lr [] -epochs []

th Hjj_Validate1.lua -data_path ActivityNet -class 1 -model_name ./New/g_1_23_fine -gpu 0 -alpha 0.2

th Hjj_Expert.lua -gpu 0 -load_memory 1 -data_path ./finetune_model/expert_experience -class 1 -model_name ./New/g_b1_14  -batch_size 200  -lr 1e-7 -epochs 30

