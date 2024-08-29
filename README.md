
[English README](README_EN.md) | [打赏项目](https://github.com/jianchang512/ChatTTS-ui/issues/122) | [Discord Discussion Group](https://discord.gg/y9gUweVCCJ)


# ChatTTS webUI & API 

一个简单的本地网页界面，在网页使用 ChatTTS 将文字合成为语音，支持中英文、数字混杂，并提供API接口.


原 [ChatTTS](https://github.com/2noise/chattts) 项目. 0.96版起，源码部署必须先安装ffmpeg ,之前的音色文件csv和pt已不可用，请填写音色值重新生成.[获取音色](?tab=readme-ov-file#音色获取)


> **[赞助商]**
> 
> [![](https://github.com/user-attachments/assets/e3e2e6f9-e2e4-44e4-860b-9d1ce5b53d4f)](https://302.ai/)
>  [302.AI](https://302.ai)是一个汇集全球顶级品牌的AI超市，按需付费，零月费，零门槛使用各种类型AI。
> 
> 功能全面、简单易用、按需付费零门槛、管理者和使用者分离
> 

**界面预览**

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/669876cf-5061-4d7d-86c5-3333d0882ee8)






文字数字符号 控制符混杂效果

https://github.com/jianchang512/ChatTTS-ui/assets/3378335/e2a08ea0-32af-4a30-8880-3a91f6cbea55


## Windows预打包版

1. 从 [Releases](https://github.com/jianchang512/chatTTS-ui/releases)中下载压缩包，解压后双击 app.exe 即可使用
2. 某些安全软件可能报毒，请退出或使用源码部署
3. 英伟达显卡大于4G显存，并安装了CUDA11.8+后，将启用GPU加速

## 手动下载模型

第一次将从huggingface.co或github下载模型到asset目录下，如果网络不稳，可能下载失败，若是失败，请单独下载

下载后解压后，会看到asset文件夹，该文件夹内有多个pt文件，将所有pt文件复制到asset目录下，然后重启软件

GitHub下载地址: https://github.com/jianchang512/ChatTTS-ui/releases/download/v1.0/all-models.7z

百度网盘下载地址: https://pan.baidu.com/s/1yGDZM9YNN7kW9e7SFo8lLw?pwd=ct5x



## Linux 下容器部署

### 安装

1. 拉取项目仓库

   在任意路径下克隆项目，例如：

   ```bash
   git clone https://github.com/jianchang512/ChatTTS-ui.git chat-tts-ui
   ```

2. 启动 Runner

   进入到项目目录：

   ```bash
   cd chat-tts-ui
   ```

   启动容器并查看初始化日志：

   ```bash
   gpu版本
   docker compose -f docker-compose.gpu.yaml up -d 

   cpu版本    
   docker compose -f docker-compose.cpu.yaml up -d

   docker compose logs -f --no-log-prefix

3. 访问 ChatTTS WebUI

   `启动:['0.0.0.0', '9966']`，也即，访问部署设备的 `IP:9966` 即可，例如：

   - 本机：`http://127.0.0.1:9966`
   - 服务器: `http://192.168.1.100:9966`

### 更新

1. Get the latest code from the main branch:

   ```bash
   git checkout main
   git pull origin main
   ```

2. Go to the next step and update to the latest image:

   ```bash
   docker compose down

   gpu版本
   docker compose -f docker-compose.gpu.yaml up -d --build

   cpu版本
   docker compose -f docker-compose.cpu.yaml up -d --build
   
   docker compose logs -f --no-log-prefix
   ```

## Linux 下源码部署

1. 配置好 python3.9-3.11环境，安装 ffmpeg。 `yum install ffmpeg` 或 `apt-get install ffmpeg`等
2. 创建空目录 `/data/chattts` 执行命令 `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`
3. 创建虚拟环境 `python3 -m venv venv`
4. 激活虚拟环境 `source ./venv/bin/activate`
5. 安装依赖 `pip3 install -r requirements.txt`
6. 如果不需要CUDA加速，执行 
	
	`pip3 install torch==2.2.0 torchaudio==2.2.0`

	如果需要CUDA加速，执行 
	
	```
	pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

	pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
		
	```
	
	另需安装 CUDA11.8+ ToolKit，请自行搜索安装方法 或参考 https://juejin.cn/post/7318704408727519270

   	除CUDA外，也可以使用AMD GPU进行加速，这需要安装ROCm和PyTorch_ROCm版本。AMG GPU借助ROCm，在PyTorch开箱即用，无需额外修改代码。
   	1. 请参考https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html 来安装AMD GPU Driver及ROCm.
	1. 再通过https://pytorch.org/ 安装PyTorch_ROCm版本。


	`pip3 install torch==2.2.0  torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm6.0`
	
    安装完成后，可以通过rocm-smi命令来查看系统中的AMD GPU。也可以用以下Torch代码(query_gpu.py)来查询当前AMD GPU Device.
	
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

 	使用以上代码，以AMD Radeon Pro W7900为例，查询设备如下。

 	```
	
 	$ python ~/query_gpu.py
	
	2.4.0.dev20240401+rocm6.0
	
 	Using GPU: AMD Radeon PRO W7900
	
 	```


 
7. 执行 `python3 app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966` (注意：默认从 modelscope 魔塔下载模型，不可使用代理下载，请关闭代理)


## MacOS 下源码部署

1. 配置好 python3.9-3.11 环境,安装git ，执行命令  `brew install libsndfile git python@3.10`
   继续执行

    ```
	brew install ffmpeg
	
    export PATH="/usr/local/opt/python@3.10/bin:$PATH"
	
    source ~/.bash_profile 
	
	source ~/.zshrc
	
    ```
	
2. 创建空目录 `/data/chattts` 执行命令 `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`
3. 创建虚拟环境 `python3 -m venv venv`
4. 激活虚拟环境 `source ./venv/bin/activate`
5. 安装依赖 `pip3 install -r requirements.txt`
6. 安装torch `pip3 install torch==2.2.0 torchaudio==2.2.0`
7. 执行 `python3 app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`  (注意：默认从 modelscope 魔塔下载模型，不可使用代理下载，请关闭代理)


## Windows源码部署

1. 下载python3.9-3.11，安装时注意选中`Add Python to environment variables`
2. 下载 ffmpeg.exe 放在 软件目录下的ffmpeg文件夹内
3. 下载并安装git，https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe 
4. 创建空文件夹 `D:/chattts` 并进入，地址栏输入 `cmd`回车，在弹出的cmd窗口中执行命令 `git clone https://github.com/jianchang512/chatTTS-ui .`
5. 创建虚拟环境，执行命令 `python -m venv venv`
6. 激活虚拟环境，执行 `.\venv\scripts\activate`
7. 安装依赖,执行 `pip install -r requirements.txt`
8. 如果不需要CUDA加速，

	执行 `pip install torch==2.2.0 torchaudio==2.2.0`

	如果需要CUDA加速，执行 
	
	`pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`
	
	另需安装 CUDA11.8+ ToolKit，请自行搜索安装方法或参考 https://juejin.cn/post/7318704408727519270
	
9. 执行 `python app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`  (注意：默认从 modelscope 魔塔下载模型，不可使用代理下载，请关闭代理)


## 源码部署注意 0.96版本起，必须安装ffmpeg

1. 如果GPU显存低于4G，将强制使用CPU。

2. Windows或Linux下如果显存大于4G并且是英伟达显卡，但源码部署后仍使用CPU，可尝试先卸载torch再重装，卸载`pip uninstall -y torch torchaudio` , 重新安装cuda版torch。`pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`  。必须已安装CUDA11.8+

3. 默认检测 modelscope 是否可连接，如果可以，则从modelscope下载模型，否则从 huggingface.co下载模型



## 音色获取

0.96版本后，因ChatTTS内核升级，已无法直接使用从该站点下载的pt文件(https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker)

因此增加转换脚本 cover-pt.py [Win整合包可以直接下载 cover-pt.exe 文件，和 app.exe 放在同一目录下双击执行](https://github.com/jianchang512/ChatTTS-ui/releases)

执行  `python cover-pt.py` 后将把 `speaker` 目录下的，以 `seed_` 开头，以  `_emb.pt` 结尾的文件，即下载后的默认文件名pt，
转换为可用的编码格式，转换后的pt将改名为以 `_emb-covert.pt` 结尾。

例：

假如  `speaker/seed_2155_restored_emb.pt` 存在这个文件,将被转换为 `speaker/seed_2155_restored_emb-cover.pt`, 然后删掉原pt文件，仅保留该转换后的文件即可





## [常见问题与报错解决方法](faq.md)




## 修改http地址

默认地址是 `http://127.0.0.1:9966`,如果想修改，可打开目录下的 `.env`文件，将 `WEB_ADDRESS=127.0.0.1:9966`改为合适的ip和端口，比如修改为`WEB_ADDRESS=192.168.0.10:9966`以便局域网可访问

## 使用API请求 v0.5+

**请求方法:** POST

**请求地址:** http://127.0.0.1:9966/tts

**请求参数:**

text:	str| 必须， 要合成语音的文字

voice:	可选，默认 2222,  决定音色的数字， 2222 | 7869 | 6653 | 4099 | 5099，可选其一，或者任意传入将随机使用音色

prompt:	str| 可选，默认 空， 设定 笑声、停顿，例如 [oral_2][laugh_0][break_6]

temperature:	float| 可选，  默认 0.3

top_p:	float|  可选， 默认 0.7

top_k:	int|  可选， 默认 20

skip_refine:	int|   可选， 默认0， 1=跳过 refine text，0=不跳过

custom_voice:	int|  可选， 默认0，自定义获取音色值时的种子值，需要大于0的整数，如果设置了则以此为准，将忽略 `voice`


**返回:json数据**

成功返回:
	{code:0,msg:ok,audio_files:[dict1,dict2]}
	
	其中 audio_files 是字典数组，每个元素dict为 {filename:wav文件绝对路径，url:可下载的wav网址}

失败返回:

	{code:1,msg:错误原因}

```

# API调用代码

import requests

res = requests.post('http://127.0.0.1:9966/tts', data={
  "text": "若不懂无需填写",
  "prompt": "",
  "voice": "3333",
  "temperature": 0.3,
  "top_p": 0.7,
  "top_k": 20,
  "skip_refine": 0,
  "custom_voice": 0
})
print(res.json())

#ok
{code:0, msg:'ok', audio_files:[{filename: E:/python/chattts/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav, url: http://127.0.0.1:9966/static/wavs/20240601-22_12_12-c7456293f7b5e4dfd3ff83bbd884a23e.wav}]}

#error
{code:1, msg:"error"}


```


## 在pyVideoTrans软件中使用

> 升级 pyVideoTrans 到 1.82+ https://github.com/jianchang512/pyvideotrans

1. 点击菜单-设置-ChatTTS，填写请求地址，默认应该填写 http://127.0.0.1:9966
2. 测试无问题后，在主界面中选择`ChatTTS`

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/7118325f-2b9a-46ce-a584-1d5c6dc8e2da)

