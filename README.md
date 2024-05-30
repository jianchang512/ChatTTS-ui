# ChatTTS webUI & API

> 这是一个绑定ChatTTS的web UI项目，提供网页中使用ChatTTS合成语音及api接口功能。
>
> 原始[ChatTTS项目](https://github.com/2noise/ChatTTS)

![image](https://github.com/jianchang512/ChatTTS-ui/assets/3378335/9477a6cf-28ac-4032-b3bc-c001f43312e5)



https://github.com/jianchang512/ChatTTS-ui/assets/3378335/b64b767c-583a-4a24-bd71-dd766144cc04



## Windows预打包版

1. 从 [Releases](https://github.com/jianchang512/chatTTS-ui/releases)中下载压缩包，解压后双击 app.exe 即可使用


## Linux 下源码部署

1. 配置好 python3.9+环境
2. 创建空目录 `/data/chattts` 执行命令 `cd /data/chattts &&  git clone https://github.com/jianchang512/chatTTS-ui .`
3. 创建虚拟环境 `python3 -m venv venv`
4. 激活虚拟环境 `source ./venv/bin/activate`
5. 安装依赖 `pip3 install -r requirements.txt`
6. 如果不需要CUDA加速，执行 `pip3 install torch torchaudio`

	如果需要CUDA加速，执行 
	```
		pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
		
		pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
	
	```
	另需安装 CUDA11.8+ ToolKit，请自行搜索安装方法
7. 执行 `python3 app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`


## MacOS 下源码部署

1. 配置好 python3.9+环境,安装git ，执行命令  `brew install git python@3.10`
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
6. 安装torch `pip3 install torch torchaudio`
7. 执行 `python3 app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`


## Windows源码部署

1. 下载python3.9+，安装时注意选中`Add Python to environment variables`
2. 下载并安装git，https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe 
3. 创建空文件夹 `D:/chattts` 并进入，地址栏输入 `cmd`回车，在弹出的cmd窗口中执行命令 `git clone https://github.com/jianchang512/chatTTS-ui .`
4. 创建虚拟环境，执行命令 `python -m venv venv`
4. 激活虚拟环境，执行 `.\venv\scripts\activate`
5. 安装依赖,执行 `pip install -r requirements.txt`
6. 如果不需要CUDA加速，执行 `pip install torch torchaudio`

	如果需要CUDA加速，执行 
	
	```
		pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
		
	```
	
	另需安装 CUDA11.8+ ToolKit，请自行搜索安装方法
7. 执行 `python app.py` 启动，将自动打开浏览器窗口，默认地址 `http://127.0.0.1:9966`


## 修改http地址

默认地址是 `http://127.0.0.1:9966`,如果想修改，可打开目录下的 `.env`文件，将 `WEB_ADDRESS=127.0.0.1:9966`改为合适的ip和端口，比如修改为`WEB_ADDRESS=192.168.0.10:9966`以便局域网可访问

## 使用API请求

请求方法:POST

请求地址: http://127.0.0.1:9966/tts

请求参数:

text:str 必须， 要合成语音的文字

voice:int 可选，  决定音色的数字， 2222 | 7869 | 6653 | 4099 | 5099，可选其一，或者任意传入将随机使用音色

prompt: str 可选，设定 笑声、停顿，例如 [oral_2][laugh_0][break_6]

返回:json数据

code=0 成功，filename=wav文件名，url=可下载的wav网址

code=1 失败，msg=错误原因

```

import requests

res=requests.post('http://127.0.0.1:9966/tts',data={"text":"你好啊亲爱的朋友。[laugh]","voice":2222,"prompt":'[oral_2][laugh_0][break_6]'})
print(res.json())

#成功
{code:0,msg:'ok',filename:1.wav,url:http://${location.host}/static/wavs/1.wav}

#error 
{code:1,msg:"error"}

```


## 在pyVideoTrans软件中使用

> 升级 pyVideoTrans 到 1.82+ https://github.com/jianchang512/pyvideotrans

1. 点击菜单-设置-ChatTTS，填写请求地址，默认应该填写 http://127.0.0.1:9966
2. 测试无问题后，在主界面中选择`ChatTTS`
