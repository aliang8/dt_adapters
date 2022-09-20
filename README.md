# DT Adapters Project

Using decision transformers as the backbone for pretraining offline a behavior policy on demonstration / play data. We then fine-tune individual adapters on top of the pretrained model for each downstream skill. To accelerate learning of compositional skills, we leverage AdapterFusion to merge several previously learned adapters together. 

# Project Setup

Setup conda env:

```
conda create -n dt_adapters python=3.7.9
```

Install metaworld 

```
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld 
pip install -e .
```

Install requirements
```
pip install -r requirements.txt
```

Some exports needed for when running mujoco_py
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin
``` 

Extra setup: installing robosuite, robomimic, etc

# TODO
[ ] Collect demonstrations for metaworld environments