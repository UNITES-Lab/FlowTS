
## the environment is exactly same as https://github.com/Y-debug-sys/Diffusion-TS, great codebase!!     
## we use os environ to pass param for: easier to understand for both human & LLM, faster to iterate, easier to remove and simplified
## You can first have a conda env, 3.8.10 is tested
# pip install -r requirements.txt    -i https://pypi.mirrors.ustc.edu.cn/simple  

## environment setting
export PATH="/opt/anaconda3/envs/torch1.8/bin:$PATH"  ## change this to your conda env, or conda activate xxx ahead. torch1.8 to your env name


## solar forecasting 
for hucfg_num_steps in 800
do
for hucfg_Kscale in 0.03  ## this has only to do with inference
do
export hucfg_attention_rope_use=-1  ## self attention will not use rope, cross attention uses 
export hucfg_lr=3e-4                ## default 1e-3
export hucfg_num_steps=${hucfg_num_steps}  ## inference steps
export hucfg_Kscale=${hucfg_Kscale}  ## k in t-power sampling
export hucfg_t_sampling=logitnorm  ## we use logit-norm as default, if want uniform sampling, change to another any name
export results_folder=./Checkpoints_solar_nips ## your checkpoints and outputs will be saved here
python solar_nips.py
done
done