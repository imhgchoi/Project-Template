# Project Template

A general project template for Deep Learning projects


## Usage

1. Replace requirements/requirements.txt with your own. Add further packages in setup.sh if needed.

2. Run requirements/setup.sh to install packages.

        chmod +x requirements/setup.sh

        requirements/setup.sh

3. Raw data for Imagenet & COCO2017 should be downloaded yourself, and its directory must be specified in the runnable shell file for use.

4. Create your shell file in run/ and execute the program. The following is an example.
        
        chmod +x run/sample.sh

        run/sample.sh

By the way, you'll need a wandb account to run the sample shell file. Check [here](https://docs.wandb.ai/quickstart) to get started with WandB!

5. For parallel computing, start off with the following command. The following is for the case when you have 8 GPU's available.

        python -m torch.distributed.launch --nproc_per_node=8 --use_env src/main.py --...