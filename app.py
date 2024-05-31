import os,sys
import torch,datetime
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import ChatTTS
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify,  send_from_directory
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
load_dotenv()
import soundfile as sf
from pathlib import Path
import hashlib,webbrowser
from modelscope import snapshot_download
import numpy as np


def get_executable_path():
    # 这个函数会返回可执行文件所在的目录
    if getattr(sys, 'frozen', False):
        # 如果程序是被“冻结”打包的，使用这个路径
        return Path(sys.executable).parent.as_posix()
    else:
        return Path.cwd().as_posix()
VERSION='0.3'

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


WEB_ADDRESS = os.getenv('WEB_ADDRESS', '127.0.0.1:9966')

# 默认从 modelscope 下载模型,如果想从huggingface下载模型，请将以下3行注释掉
CHATTTS_DIR = snapshot_download('pzc163/chatTTS',cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local",local_path=CHATTTS_DIR)

# 如果希望从 huggingface.co下载模型，将以下注释删掉。将上方3行内容注释掉
#os.environ['HF_HUB_CACHE']=MODEL_DIR
#os.environ['HF_ASSETS_CACHE']=MODEL_DIR
#chat = ChatTTS.Chat()
#chat.load_models()




# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

app = Flask(__name__, static_folder=ROOT_DIR+'/static', static_url_path='/static',
            template_folder=ROOT_DIR+'/templates')

root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.WARNING)
app.logger.setLevel(logging.WARNING) 
# 创建 RotatingFileHandler 对象，设置写入的文件路径和大小限制
file_handler = RotatingFileHandler(LOGS_DIR+f'/{datetime.datetime.now().strftime("%Y%m%d")}.log', maxBytes=1024 * 1024, backupCount=5)
# 创建日志的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置文件处理器的级别和格式
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
# 将文件处理器添加到日志记录器中
app.logger.addHandler(file_handler)
app.jinja_env.globals.update(enumerate=enumerate)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


@app.route('/')
def index():
    return render_template("index.html",weburl=WEB_ADDRESS,version=VERSION)


# 根据文本返回tts结果，返回 filename=文件名 url=可下载地址
# 请求端根据需要自行选择使用哪个
# params
# text:待合成文字
# voice：音色
# custom_voice：自定义音色值
# skip_refine: 1=跳过refine_text阶段，0=不跳过
# prompt：
@app.route('/tts', methods=['GET', 'POST'])
def tts():
    # 原始字符串
    text = request.args.get("text","").strip() or request.form.get("text","").strip()
    prompt = request.form.get("prompt",'')
    try:
        custom_voice=request.form.get("custom_voice",'')
        voice = int(custom_voice) if custom_voice else int(request.form.get("voice",'2222'))
    except Exception:
        voice=2222
    
    skip_refine = request.form.get("skip_refine",'0')
    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})
    std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
    torch.manual_seed(voice)

    rand_spk = chat.sample_random_speaker()
    #rand_spk = torch.randn(768) * std + mean

    md5_hash = hashlib.md5()
    md5_hash.update(f"{text}-{voice}-{skip_refine}-{prompt}".encode('utf-8'))
    datename=datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
    filename = datename+'-'+md5_hash.hexdigest() + ".wav"
    if int(skip_refine)==1:
        wavs = chat.infer([t for t in text.split("\n") if t.strip()], use_decoder=True, skip_refine_text=True,params_infer_code={'spk_emb': rand_spk}, params_refine_text= {'prompt': prompt})
    else:
        wavs = chat.infer([t for t in text.split("\n") if t.strip()], use_decoder=True, params_infer_code={'spk_emb': rand_spk}, params_refine_text= {'prompt': prompt})
    # 初始化一个空的numpy数组用于之后的合并
    combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # 确保dtype与你的wav数据类型匹配

    for wavdata in wavs:
        combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))

    sf.write(WAVS_DIR+'/'+filename, combined_wavdata, 24000)
    return jsonify({"code": 0, "msg": "ok","filename":WAVS_DIR+'/'+filename,"url":f"http://{WEB_ADDRESS}/static/wavs/{filename}"})



try:
    host = WEB_ADDRESS.split(':')
    print(f'启动:{host}')
    webbrowser.open(f'http://{WEB_ADDRESS}')
    serve(app,host=host[0], port=int(host[1]))
except Exception:
    pass

