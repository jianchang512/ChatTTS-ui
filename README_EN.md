[中文](README.md)

# ChatTTS WebUI & API 

A simple local web interface to convert text to speech using ChatTTS. It supports mixed Chinese, English, and numbers, and provides an API.

> Original [ChatTTS](https://github.com/2noise/chattts) project

**UI Preview**

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/8d9b36d4-29b9-4cd7-ae70-3e3bd3225108)


> The model will be downloaded automatically on the first launch. It will try connecting to `https://huggingface.co` first. If unreachable, it will download from ModelScope (modelscope.cn).
>
> Please ensure you have a stable internet connection (a VPN may be needed in some regions) when deploying from source to avoid download failures.

## Windows Pre-packaged Version

1. Download the zip file from [Releases](https://github.com/jianchang512/chatTTS-ui/releases), extract it, and double-click `app.exe` to start.
2. Some antivirus software may trigger false positives. You can temporarily disable them or deploy from source instead.
3. GPU acceleration will be enabled if you have an NVIDIA card with more than 4GB VRAM and CUDA 12.8+ installed.

## Linux Source Deployment

1. Set up a Python 3.9–3.11 environment.
2. Create an empty directory `/data/chattts` and run: `cd /data/chattts && git clone https://github.com/jianchang512/chatTTS-ui .`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate the virtual environment: `source ./venv/bin/activate`
5. Install dependencies: `pip3 install -r requirements.txt`
6. For CPU-only (no CUDA acceleration), run: 
	
	`pip3 install torch==2.7.1 torchaudio==2.7.1`

   For GPU acceleration (CUDA), run: 
	
	```bash
	pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
	pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
	```
	
	You will also need to install the CUDA 12.8+ Toolkit yourself.
	
7. Run `python3 app.py` to start. A browser window will open automatically at the default address `http://127.0.0.1:9966`. (Note: By default, the model is downloaded from ModelScope. Please turn off your proxy/VPN during this download, as proxies are not supported for ModelScope).

## macOS Source Deployment

1. Set up a Python 3.9–3.11 environment, install Git, and run: `brew install libsndfile git python@3.10`
   Then run:

    ```bash
    export PATH="/usr/local/opt/python@3.10/bin:$PATH"
    source ~/.bash_profile 
    source ~/.zshrc
    ```
	
2. Create an empty directory `/data/chattts` and run: `cd /data/chattts && git clone https://github.com/jianchang512/chatTTS-ui .`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate the virtual environment: `source ./venv/bin/activate`
5. Install dependencies: `pip3 install -r requirements.txt`
6. Install Torch: `pip3 install torch==2.7.1 torchaudio==2.7.1`
7. Run `python3 app.py` to start. A browser window will open automatically at `http://127.0.0.1:9966`.

## Windows Source Deployment

1. Download and install Python 3.9–3.11. Make sure to check "Add Python to environment variables" during installation.
2. Download and install Git: https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe 
3. Create an empty folder named `D:/chattts`. Open the folder, type `cmd` in the address bar, press Enter, and run: `git clone https://github.com/jianchang512/chatTTS-ui .`
4. Create a virtual environment: `python -m venv venv`
5. Activate the virtual environment: `.\venv\Scripts\activate`
6. Install dependencies: `pip install -r requirements.txt`
7. For CPU-only (no CUDA acceleration):

	Run `pip install torch==2.7.1 torchaudio==2.7.1`

	For GPU acceleration (CUDA), run: 
	
	`pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128`
	
	You will also need to install the CUDA 12.8+ Toolkit yourself.
	
8. Run `python app.py` to start. A browser window will open automatically at `http://127.0.0.1:9966`. (Note: By default, the model is downloaded from ModelScope. Please turn off your proxy/VPN during this download).

## Docker Deployment (Linux)

### Installation

1. Clone the repository

   Clone the project into any directory, for example:

   ```bash
   git clone https://github.com/jianchang512/ChatTTS-ui.git chat-tts-ui
   ```

2. Start the Container

   Navigate to the project directory:

   ```bash
   cd chat-tts-ui
   ```

   Start the container and view the initialization logs:

   ```bash
   # GPU version
   docker compose -f docker-compose.gpu.yaml up -d 

   # CPU version    
   docker compose -f docker-compose.cpu.yaml up -d

   docker compose logs -f --no-log-prefix
   ```

3. Access ChatTTS WebUI

   Once started on `0.0.0.0:9966`, you can access the WebUI using the device's `IP:9966`. For example:

   - Local: `http://127.0.0.1:9966`
   - Server: `http://192.168.1.100:9966`

### Updating

1. Get the latest code from the main branch:

   ```bash
   git checkout main
   git pull origin main
   ```

2. Stop the container and rebuild with the latest image:

   ```bash
   docker compose down

   # GPU version
   docker compose -f docker-compose.gpu.yaml up -d --build

   # CPU version
   docker compose -f docker-compose.cpu.yaml up -d --build
   
   docker compose logs -f --no-log-prefix
   ```

## Deployment Notes

1. If your GPU VRAM is less than 4GB, CPU mode will be forced.

2. On Windows or Linux, if you have an NVIDIA card with >4GB VRAM but the software still uses CPU, try uninstalling and reinstalling Torch. Run `pip uninstall -y torch torchaudio` first, then reinstall the CUDA version: `pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128`. Ensure you have CUDA 12.8+ installed.

3. The software checks if ModelScope is accessible. If yes, it downloads the model from ModelScope; otherwise, it downloads from huggingface.co.

## Obtaining Voices (Speakers)

Starting from version 0.92, fixed voices in `.csv` or `.pt` format are supported. Simply download and save them in the `speaker` folder inside the software directory.

`.pt` files can be downloaded from the demo link page of the [ChatTTS_Speaker](https://github.com/6drf21e/ChatTTS_Speaker) project (https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker).

You can also visit http://ttslist.aiqbh.com/10000cn/ to listen to audio samples and copy the corresponding voice seed value into the "Custom Voice Value" text box.

**Note:** The same voice seed value may sound slightly different across different devices. Even on the same device, the tone or pitch of the same voice seed may vary slightly between generations.

## Adjusting Environment Settings

Open the `.env` file with a text editor to modify the configuration:

```ini
WEB_ADDRESS=127.0.0.1:9966 # Modify the Web service address and port
compile=false 
device=default # Defaults to CUDA (if VRAM > 4GB) or MPS. You can manually set this to cpu, mps, or cuda
endpoint=https://hf-mirror.com # Model download source. Defaults to a mirror. Change to https://huggingface.co if accessible.
```

## [FAQ & Troubleshooting](faq.md)

## Modifying the HTTP Address

The default address is `http://127.0.0.1:9966`. To change this, open the `.env` file in the project folder and change `WEB_ADDRESS=127.0.0.1:9966` to your preferred IP and port (e.g., `WEB_ADDRESS=192.168.0.10:9966` to allow local network access).

## Using the API (v0.5+)

**Request Method:** POST

**Request URL:** http://127.0.0.1:9966/tts

**Parameters:**

- `text`: str | Required. The text to be synthesized.
- `voice`: Optional. Default is 2222. Numeric seed for the voice (e.g., 2222, 7869, 6653, 4099, 5099). Pass any number for a random voice.
- `prompt`: str | Optional. Default is empty. Used to insert laughter or pauses, e.g., `[oral_2][laugh_0][break_6]`.
- `temperature`: float | Optional. Default is 0.3.
- `top_p`: float | Optional. Default is 0.7.
- `top_k`: int | Optional. Default is 20.
- `skip_refine`: int | Optional. Default is 0. `1` = skip text refinement, `0` = do not skip.
- `custom_voice`: int | Optional. Default is 0. Seed value for custom voice. Must be an integer greater than 0. If set, this overrides the `voice` parameter.

**Response: JSON**

Success:
```json
{
  "code": 0,
  "msg": "ok",
  "audio_files": [
    {
      "filename": "absolute path to WAV",
      "url": "downloadable URL for WAV"
    }
  ]
}
```
	
Failure:
```json
{
  "code": 1,
  "msg": "error details"
}
```

```python
# API Code Example

import requests

res = requests.post('http://127.0.0.1:9966/tts', data={
  "text": "Hello world.",
  "prompt": "",
  "voice": "3333",
  "temperature": 0.3,
  "top_p": 0.7,
  "top_k": 20,
  "skip_refine": 0,
  "custom_voice": 0
})
print(res.json())

# Success Response
# {code:0, msg:'ok', audio_files:[{filename: E:/python/chattts/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav, url: http://127.0.0.1:9966/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav}]}

# Error Response
# {code:1, msg:"error"}
```

## Using with pyVideoTrans

> Upgrade pyVideoTrans to 1.82+ (https://github.com/jianchang512/pyvideotrans)

1. Click Menu -> Settings -> ChatTTS, and enter the request URL (default is `http://127.0.0.1:9966`).
2. Once the test is successful, select `ChatTTS` in the main interface.

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/7118325f-2b9a-46ce-a584-1d5c6dc8e2da)