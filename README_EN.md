
[简体中文](README.md) | [Discord Discussion Group](https://discord.gg/y9gUweVCCJ) | [Support the Project](https://github.com/jianchang512/ChatTTS-ui/issues/122)

# ChatTTS webUI & API 

A simple local web interface to use ChatTTS for text-to-speech synthesis on the web, supporting mixed Chinese and English text and numbers, and providing an API interface.

> The original [ChatTTS](https://github.com/2noise/chattts) project

**Interface Preview**

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/8d9b36d4-29b9-4cd7-ae70-3e3bd3225108)


Sample synthesized voice effects

https://github.com/jianchang512/ChatTTS-ui/assets/3378335/bd6aaef9-a49a-4a81-803a-91e3320bf808

Text and control symbols mixed effect

https://github.com/jianchang512/ChatTTS-ui/assets/3378335/e2a08ea0-32af-4a30-8880-3a91f6cbea55


## Windows Pre-packaged Version

1. Download the compressed package from [Releases](https://github.com/jianchang512/chatTTS-ui/releases), unzip it, and double-click app.exe to use.
2. Some security software may flag it as a virus, please disable or deploy from source.
3. If you have an Nvidia graphics card with more than 4GB of memory and have installed CUDA11.8+, GPU acceleration will be enabled.

## Linux Container Deployment

### Installation

1. Clone the project repository

   Clone the project to any directory, for example:

   ```bash
   git clone https://github.com/jianchang512/ChatTTS-ui.git chat-tts-ui
   ```

2. Start Runner

   Enter the project directory:

   ```bash
   cd chat-tts-ui
   ```

   Start the container and view the initialization logs:

   ```bash
   For GPU version
   docker compose -f docker-compose.gpu.yaml up -d 

   For CPU version    
   docker compose -f docker-compose.cpu.yaml up -d

   docker compose logs -f --no-log-prefix
   ```

3. Access ChatTTS WebUI

   `Started at:['0.0.0.0', '9966']`, meaning you can access it via `IP:9966` of the deployment device, for example:

   - Localhost: `http://127.0.0.1:9966`
   - Server: `http://192.168.1.100:9966`

### Update

1. Get the latest code from the main branch:

   ```bash
   git checkout main
   git pull origin main
   ```

2. Go to the next step and update to the latest image:

   ```bash
   docker compose down

   For GPU version
   docker compose -f docker-compose.gpu.yaml up -d --build

   For CPU version
   docker compose -f docker-compose.cpu.yaml up -d --build
   
   docker compose logs -f --no-log-prefix
   ```

## Linux Source Code Deployment

1. Prepare python3.9-3.11 environment.
2. Create an empty directory `/data/chattts` and execute `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`.
3. Create a virtual environment `python3 -m venv venv`.
4. Activate the virtual environment `source ./venv/bin/activate`.
5. Install dependencies `pip3 install -r requirements.txt`.
6. If CUDA acceleration is not needed, execute 

   `pip3 install torch==2.2.0 torchaudio==2.2.0`

   If CUDA acceleration is needed, execute
   
   ```
   pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

   pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
   ```
   
   Additionally, install CUDA11.8+ ToolKit, search for installation methods or refer to https://juejin.cn/post/7318704408727519270

   Besides CUDA, AMD GPU acceleration can also be used by installing ROCm and PyTorch_ROCm version. For AMD GPU, with the help of ROCm, PyTorch works out of the box without further modifications.
   1. Refer to https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html to install AMD GPU Driver and ROCm.
   2. Then install PyTorch_ROCm version from https://pytorch.org/. 

   `pip3 install torch==2.2.0  torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm6.0`

   After installation, you can use the command `rocm-smi` to view the AMD GPUs in the system. The following Torch code(query_gpu.py) can also be used to query the current AMD GPU Device.

   ```
   import torch
   
   print(torch.__version__)
   
   if torch.cuda.is_available():
       device = torch.device("cuda")          # a CUDA device object
       print('Using GPU:', torch.cuda.get_device_name(0))
   else:
       device = torch.device("cpu")
       print('Using CPU')
   
   torch.cuda.get_device_properties(0)

   ```

   Using the code above, for instance, with AMD Radeon Pro W7900, the device query is as follows.

   ```
   
   $ python ~/query_gpu.py
   
   2.4.0.dev20240401+rocm6.0
   
   Using GPU: AMD Radeon PRO W7900
   
   ```


 
7. Execute `python3 app.py` to start. It will automatically open a browser window at `http://127.0.0.1:9966`. Note: Models are downloaded from modelscope by default without using a proxy, please disable the proxy.


## MacOS Source Code Deployment

1. Prepare the python3.9-3.11 environment and install git. Execute command `brew install libsndfile git python@3.10`. Then continue with

   ```
   export PATH="/usr/local/opt/python@3.10/bin:$PATH"
   
   source ~/.bash_profile 
   
   source ~/.zshrc
   
   ```
   
2. Create an empty directory `/data/chattts` and execute command `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`.
3. Create a virtual environment `python3 -m venv venv`.
4. Activate the virtual environment `source ./venv/bin/activate`.
5. Install dependencies `pip3 install -r requirements.txt`.
6. Install torch `pip3 install torch==2.2.0 torchaudio==2.2.0`.
7. Execute `python3 app.py` to start. It will automatically open a browser window at `http://127.0.0.1:9966`. Note: Models are downloaded from modelscope by default without using a proxy, please disable the proxy.


## Windows Source Code Deployment

1. Download python3.9-3.11, make sure to check `Add Python to environment variables` during installation.
2. Download and install git from https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe.
3. Create an empty folder `D:/chattts` and enter it, type `cmd` in the address bar and press Enter. In the cmd window that pops up, execute command `git clone https://github.com/jianchang512/chatTTS-ui .`.
4. Create a virtual environment by executing command `python -m venv venv`.
5. Activate the virtual environment by executing `.\venv\scripts\activate`.
6. Install dependencies by executing `pip install -r requirements.txt`.
7. If CUDA acceleration is not needed,

   execute `pip install torch==2.2.0 torchaudio==2.2.0`.

   If CUDA acceleration is needed, execute 
   
   `pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`.
   
   Additionally, install CUDA11.8+ ToolKit, search for installation methods or refer to https://juejin.cn/post/7318704408727519270.
   
8. Execute `python app.py` to start. It will automatically open a browser window at `http://127.0.0.1:9966`. Note: Models are downloaded from modelscope by default without using a proxy, please disable the proxy.


## Deployment Notes

1. If the GPU memory is below 4GB, it will forcefully use the CPU.

2. Under Windows or Linux, if the memory is more than 4GB and it is an Nvidia graphics card, but the source code deployment still uses CPU, you may try uninstalling torch first and then reinstalling it. Uninstall with `pip uninstall -y torch torchaudio`, then reinstall the CUDA version of torch `pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`. CUDA11.8+ must be installed.

3. By default, it checks whether modelscope can be connected. If so, models are downloaded from modelscope; otherwise, models are downloaded from huggingface.co.


## [FAQs and Troubleshooting](faq.md)




## Modify HTTP Address

The default address is `http://127.0.0.1:9966`. If you want to modify it, open the `.env` file in the directory and change `WEB_ADDRESS=127.0.0.1:9966` to the appropriate IP and port, such as changing to `WEB_ADDRESS=192.168.0.10:9966` for LAN access.

## Using API Requests v0.5+

**Method:** POST

**URL:** http://127.0.0.1:9966/tts

**Parameters:**

text: str| Required, text to synthesize.

voice: int| Optional, default 2222. Determines the voice digit, choose from 2222 | 7869 | 6653 | 4099 | 5099, or any input will randomly use a voice.

prompt: str| Optional, default empty. Sets laughter, pause, etc., like [oral_2][laugh_0][break_6].

temperature: float| Optional, default 0.3.

top_p: float| Optional, default 0.7.

top_k: int| Optional, default 20.

skip_refine: int| Optional, default 0. 1=skip refine text, 0=do not skip.

custom_voice: int| Optional, default 0. Sets a custom seed value for obtaining the voice, must be a positive integer. If set, it will take precedence over `voice`.


**Response: JSON**

Success:
	{code:0,msg:ok,audio_files:[dict1,dict2]}
	
	where audio_files is an array of dictionaries, each element dict is {filename:absolute path to wav file, url:downloadable wav URL}

Failure:

	{code:1,msg:error reason}


```

# API Call Code

import requests

res = requests.post('http://127.0.0.1:9966/tts', data={
  "text": "No need to fill if unsure",
  "prompt": "",
  "voice": "3333",
  "temperature": 0.3,
  "top_p": 0.7,
  "top_k": 20,
  "skip_refine": 0,
  "custom_voice": 0,
})
print(res.json())

#ok
{code:0, msg:'ok', audio_files:[{filename: E:/python/chattts/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav, url: http://127.0.0.1:9966/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav}]}

#error
{code:1, msg:"error"}


```


## Using in pyVideoTrans software

> Upgrade pyVideoTrans to 1.82+ https://github.com/jianchang512/pyvideotrans

1. Click Menu-Settings-ChatTTS and fill in the request address, which should by default be http://127.0.0.1:9966.
2. After ensuring there are no issues, select `ChatTTS` on the main interface.

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/7118325f-2b9a-46ce-a584-1d5c6dc8e2da)

