# Build a question answering system for both bangla and english language using any open source LLM and fine tune it for My own data

# Project Overview
This project aims to set up a Python environment, install necessary libraries for machine learning, and demonstrate a simple use case involving loading and processing a dataset. This project uses tools such as torch, transformers, bitsandbytes, and others.

# Prerequisites
Ubuntu or a compatible Linux distribution
Python 3.10 or higher

# Setup Instructions
1. Install Python Virtual Environment
To isolate the project dependencies, we will use Python's virtual environment tool.

bash
!apt install python3.10-venv
This command installs the Python 3.10 virtual environment package on your system.

# 2. Create a Virtual Environment
Create a virtual environment named myenv.

bash
!python -m venv myenv
!python -m venv myenv
The venv module creates a virtual environment named myenv in your current directory.

# 3. Activate the Virtual Environment
Activate the newly created virtual environment.

bash
!source myenv/bin/activate
This command activates the virtual environment, which isolates your project’s dependencies from your system’s Python environment.

# 4. Upgrade pip
Ensure you have the latest version of pip.

bash
pip install --upgrade pip
This command upgrades pip to the latest version available.

# 5. Install PyTorch and Other Packages
Install specific versions of torch, torchaudio, and torchvision.

bash
!pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
These commands install the specified versions of torch, torchaudio, and torchvision.

# 6. Install Hugging Face Libraries from GitHub
Install specific versions of Hugging Face transformers, peft, and accelerate directly from their GitHub repositories.

bash
!pip install -U git+https://github.com/huggingface/transformers.git@e03a9cc
!pip install -U git+https://github.com/huggingface/peft.git@42a184f
!pip install -U git+https://github.com/huggingface/accelerate.git@c9fbb71
These commands install the specified commits of the libraries from their GitHub repositories.

# 7. Install Additional Libraries
Install datasets, loralib, and einops.

bash
!pip install datasets==2.12.0 loralib==0.1.1 einops==0.6.1
This command installs the specified versions of datasets, loralib, and einops.

# 8. Import Libraries
Import the necessary libraries for your project.

python
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
These imports include essential packages for data manipulation, machine learning, and Hugging Face model handling.

# 9. Configure CUDA Devices
     Set the CUDA visible devices.

python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
This command ensures that your code uses the GPU at index 0.

# 10. Download a File from Google Drive
Use gdown to download a file from Google Drive.

bash
!gdown 1u85RQZdRTmpjGKcCc5anCMAHZ-um4DUC
This command downloads a file from Google Drive using its ID.

# 11. Load and Display JSON Data
Load data from a JSON file and print the first question.

python
with open("ecommerce-faq.json") as json_file:
    data = json.load(json_file)
pprint(data["questions"][0], sort_dicts=False)
These commands load the JSON file and print the first question in the dataset.

# License
This project is licensed under the MIT License.
