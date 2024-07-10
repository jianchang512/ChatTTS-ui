'''
0.96版本后，因ChatTTS内核升级，已无法直接使用从该站点下载的pt文件。

https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker

因此增加该转换脚本。

执行  python cover-pt.py 后将把 `speaker` 目录下的，以 seed_ 开头，
以  _emb.pt 结尾的文件，即下载后的默认文件名，
转换为可用的编码格式，转换后的pt将改名为以 `_emb-covert.pt` 结尾。

例：

假如  speaker/seed_2155_restored_emb.pt 存在这个文件

将被转换为 speaker/seed_2155_restored_emb-cover.pt, 

然后删掉原pt文件，仅保留该转换后的文件即可



'''

import os
import re
import sys
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import io
import json
import torchaudio
import wave
from pathlib import Path
print('Starting...')
import shutil
import time


import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import subprocess
import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv
load_dotenv()

import logging
from logging.handlers import RotatingFileHandler

from random import random
from modelscope import snapshot_download
import numpy as np
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils,VERSION
from ChatTTS.utils.gpu_utils import select_device
from uilib.utils import is_chinese_os,modelscope_status
merge_size=int(os.getenv('merge_size',10))
env_lang=os.getenv('lang','')
if env_lang=='zh':
    is_cn= True
elif env_lang=='en':
    is_cn=False
else:
    is_cn=is_chinese_os()
    
CHATTTS_DIR= MODEL_DIR+'/pzc163/chatTTS'


chat = ChatTTS.Chat()
device=os.getenv('device','default')
chat.load(source="custom",custom_path=CHATTTS_DIR, device=None if device=='default' else device,compile=True if os.getenv('compile','true').lower()!='false' else False)
n=0
for it in os.listdir('./speaker'):
    if it.startswith('seed_') and not it.endswith('_emb-covert.pt'):
        print(f'开始转换 {it}')        
        n+=1
        rand_spk=torch.load(f'./speaker/{it}', map_location=select_device(4096) if device=='default' else torch.device(device))

        torch.save( chat._encode_spk_emb(rand_spk) ,f"{SPEAKER_DIR}/{it.replace('.pt','-covert.pt')}")
if n==0:
    print('没有可转换的pt文件，仅转换以 seed_ 开头，并以 _emb.pt 结尾的文件')

else:
    print(f'转换完成{n}个，可以删掉以 _emb.pt 结尾的文件了，注意保留 -covert.pt 结尾的文件')

print(f'\n\n30s后本窗口自动关闭')
time.sleep(30)