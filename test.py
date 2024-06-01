import os,sys
from pathlib import Path

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
os.environ['HF_HUB_CACHE']=MODEL_DIR
os.environ['HF_ASSETS_CACHE']=MODEL_DIR
import ChatTTS
import torch,datetime
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import soundfile as sf
import numpy as np







WAVS_DIR_PATH=Path(ROOT_DIR+"/static/wavs")
WAVS_DIR_PATH.mkdir(parents=True, exist_ok=True)
WAVS_DIR=WAVS_DIR_PATH.as_posix()

LOGS_DIR_PATH=Path(ROOT_DIR+"/logs")
LOGS_DIR_PATH.mkdir(parents=True, exist_ok=True)
LOGS_DIR=LOGS_DIR_PATH.as_posix()




# cong huggingface 下载模型
chat = ChatTTS.Chat()
chat.load_models(force_redownload=True)

exit()

text="你好啊朋友们,听说今天是个好日子,难道不是吗？"
prompt='[oral_2][laugh_0][break_0]'
#
torch.manual_seed(3333)
rand_spk = chat.sample_random_speaker()
print(rand_spk)

wavs = chat.infer([text], use_decoder=True,params_infer_code={'spk_emb': rand_spk,'prompt':'[speed_1]'} ,skip_refine_text=True,params_refine_text= {'prompt': prompt})
# 初始化一个空的numpy数组用于之后的合并
combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # 确保dtype与你的wav数据类型匹配

for wavdata in wavs:
    combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))
sf.write('test.wav', combined_wavdata, 24000)

