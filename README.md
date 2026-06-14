
[English README](README_EN.md)


# ChatTTS webUI & API 

一个简单的本地网页界面，在网页使用 ChatTTS 将文字合成为语音，支持中英文、数字混杂，并提供API接口。

> 原始[ChatTTS](https://github.com/2noise/chattts)项目



**界面预览**


![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/669876cf-5061-4d7d-86c5-3333d0882ee8)



试听合成语音效果

https://github.com/jianchang512/ChatTTS-ui/assets/3378335/bd6aaef9-a49a-4a81-803a-91e3320bf808


文字数字符号 控制符混杂效果

https://github.com/jianchang512/ChatTTS-ui/assets/3378335/e2a08ea0-32af-4a30-8880-3a91f6cbea55


> 第一次启动时将在校下载模型，先检测能否连接  `https://huggingface.co`, 若不可连接，则从阿里魔塔 modelscope.cn 下载
>
> 源码部署时请确保科学上网，否则下载模型会失败

## Windows预打包版

1. 从 [Releases](https://github.com/jianchang512/chatTTS-ui/releases)中下载压缩包，解压后双击 app.exe 即可使用
2. 某些安全软件可能报毒，请退出或使用源码部署
3. 英伟达显卡大于4G显存，并安装了CUDA12.8+后，将启用GPU加速


## Linux 下源码部署

1. 配置好 python3.9-3.11环境
2. 创建空目录 `/data/chattts` 执行命令 `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`
3. 创建虚拟环境 `python3 -m venv venv`
4. 激活虚拟环境 `source ./venv/bin/activate`
5. 安装依赖 `pip3 install -r requirements.txt`
6. 如果不需要CUDA加速，执行 
	
	`pip3 install torch==2.7.1 torchaudio==2.7.1`

	如果需要CUDA加速，执行 
	
	```
	pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

	pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
		
	```
	
	另需安装 CUDA12.8+ ToolKit，请自行搜索安装方法
	

 
7. 执行 `python3 app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966` (注意：默认从 modelscope 魔塔下载模型，不可使用代理下载，请关闭代理)


## MacOS 下源码部署

1. 配置好 python3.9-3.11 环境,安装git ，执行命令  `brew install libsndfile git python@3.10`
   继续执行

    ```
    export PATH="/usr/local/opt/python@3.10/bin:$PATH"
	
    source ~/.bash_profile 
	
	source ~/.zshrc
	
    ```
	
2. 创建空目录 `/data/chattts` 执行命令 `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`
3. 创建虚拟环境 `python3 -m venv venv`
4. 激活虚拟环境 `source ./venv/bin/activate`
5. 安装依赖 `pip3 install -r requirements.txt`
6. 安装torch `pip3 install torch==2.7.1 torchaudio==2.7.1`
7. 执行 `python3 app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`


## Windows源码部署

1. 下载python3.9-3.11，安装时注意选中`Add Python to environment variables`
2. 下载并安装git，https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe 
3. 创建空文件夹 `D:/chattts` 并进入，地址栏输入 `cmd`回车，在弹出的cmd窗口中执行命令 `git clone https://github.com/jianchang512/chatTTS-ui .`
4. 创建虚拟环境，执行命令 `python -m venv venv`
4. 激活虚拟环境，执行 `.\venv\scripts\activate`
5. 安装依赖,执行 `pip install -r requirements.txt`
6. 如果不需要CUDA加速，

	执行 `pip install torch==2.7.1 torchaudio==2.7.1`

	如果需要CUDA加速，执行 
	
	`pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128`
	
	另需安装 CUDA12.8+ ToolKit，请自行搜索安装方法
	
7. 执行 `python app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`  (注意：默认从 modelscope 魔塔下载模型，不可使用代理下载，请关闭代理)

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



## 部署注意

1. 如果GPU显存低于4G，将强制使用CPU。

2. Windows或Linux下如果显存大于4G并且是英伟达显卡，但源码部署后仍使用CPU，可尝试先卸载torch再重装，卸载`pip uninstall -y torch torchaudio` , 重新安装cuda版torch。`pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128`  。必须已安装CUDA12.8+

3. 默认检测 modelscope 是否可连接，如果可以，则从modelscope下载模型，否则从 huggingface.co下载模型

## 音色获取

从 0.92 版本起，支持csv或pt格式的固定音色，下载后保存到软件目录下的  speaker 文件夹中即可

pt文件可从 https://github.com/6drf21e/ChatTTS_Speaker 项目提供的体验链接页面 (https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker) 下载。

也可以从此页面 http://ttslist.aiqbh.com/10000cn/  查看试听后将对应音色值填写到 “自定义音色值”文本框中

**不同设备同一音色值seed，最终合成的声音会有差异的，以及同一设备相同音色值，音色也可能会有变化，尤其音调**


## 环境调整
记事本打开 .env 文件，可看到以下代码
```
WEB_ADDRESS=127.0.0.1:9966 #可修改web服务地址和端口
compile=false 
device=default # 默认优先选择cuda(若显存大于4G)/mps，可手动指定 cpu|mps|cuda
endpoint=https://hf-mirror.com # 模型下载地址，默认使用国内镜像，若可访问hf，可修改为 https://huggingface.co


```



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

