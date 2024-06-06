import os
import sys,re
from pathlib import Path

import re
from uilib import utils
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

def get_executable_path():
    # 这个函数会返回可执行文件所在的目录
    if getattr(sys, 'frozen', False):
        # 如果程序是被“冻结”打包的，使用这个路径
        return Path(sys.executable).parent.as_posix()
    else:
        return Path.cwd().as_posix()

ROOT_DIR=get_executable_path()

MODEL_DIR_PATH=Path(ROOT_DIR+"/models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR=MODEL_DIR_PATH.as_posix()

WAVS_DIR_PATH=Path(ROOT_DIR+"/static/wavs")
WAVS_DIR_PATH.mkdir(parents=True, exist_ok=True)
WAVS_DIR=WAVS_DIR_PATH.as_posix()

LOGS_DIR_PATH=Path(ROOT_DIR+"/logs")
LOGS_DIR_PATH.mkdir(parents=True, exist_ok=True)
LOGS_DIR=LOGS_DIR_PATH.as_posix()

import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
load_dotenv()


import hashlib,webbrowser
from modelscope import snapshot_download
import numpy as np
import time
# 读取 .env 变量
WEB_ADDRESS = os.getenv('WEB_ADDRESS', '127.0.0.1:9966')

# 默认从 modelscope 下载模型,如果想从huggingface下载模型，请将以下3行注释掉
CHATTTS_DIR = snapshot_download('pzc163/chatTTS',cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local",local_path=CHATTTS_DIR,compile=True if os.getenv('compile','true').lower()!='false' else False)



# 如果希望从 huggingface.co下载模型，将以下注释删掉。将上方3行内容注释掉
#os.environ['HF_HUB_CACHE']=MODEL_DIR
#os.environ['HF_ASSETS_CACHE']=MODEL_DIR
#chat = ChatTTS.Chat()
#chat.load_models(compile=True if os.getenv('compile','true').lower()!='false' else False)




text="我有12879651325.68元钱[laugh][laugh]，占全部幻想的56.2%，我的手机号码是12312345678，[laugh]座机是0532-84752563，[1break][1break]现在是2013-5-1，12:14:13计算1+2=3，[uv_break][uv_break]6*7=42？"

prompt='[oral_2][laugh_0][break_0]'
#
torch.manual_seed(1111)
rand_spk = chat.sample_random_speaker()


wavs = chat.infer([text], use_decoder=True,do_text_normalization=False,skip_refine_text=True)
# 初始化一个空的numpy数组用于之后的合并
combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # 确保dtype与你的wav数据类型匹配

for wavdata in wavs:
    combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))
sf.write('1111.wav', combined_wavdata, 24000)

