import os
import re
import sys
from dotenv import load_dotenv
load_dotenv()
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import io
import json
import wave
from pathlib import Path
print('Starting...')
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
import soundfile as sf
import ChatTTS
import datetime
from flask import Flask, request, render_template, jsonify,  send_from_directory,send_file,Response, stream_with_context
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
import huggingface_hub
from huggingface_hub.errors import LocalEntryNotFoundError
from random import random
from modelscope import snapshot_download
import numpy as np
import time
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils,VERSION
from ChatTTS.utils.gpu_utils import select_device
from uilib.utils import is_chinese_os,modelscope_status,is_connect_hf
env_lang=os.getenv('lang','')
if env_lang=='zh':
    is_cn= True
elif env_lang=='en':
    is_cn=False
else:
    is_cn=is_chinese_os()
    
CHATTTS_DIR= MODEL_DIR+'/pzc163/chatTTS'
os.environ['HF_HUB_CACHE']=MODEL_DIR
os.environ['HF_ASSETS_CACHE']=MODEL_DIR


try:
    # 优先使用本地模型，不存在再联网检测下载
    huggingface_hub.snapshot_download(
                repo_id="2Noise/ChatTTS",
                local_dir=CHATTTS_DIR,
                etag_timeout=5,
                local_files_only=True
            )
except LocalEntryNotFoundError:
    if not is_connect_hf():
        print(f"当前从阿里魔塔下载模型...\n")
        os.environ['HF_ENDPOINT']="https://hf-mirror.com"
        from modelscope import snapshot_download
        snapshot_download(model_id='pzc163/chatTTS',local_dir=CHATTTS_DIR)
    else:
        print(f"当前从 huggingface.co 下载模型...\n")
        huggingface_hub.snapshot_download(local_dir=CHATTTS_DIR,repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"],local_files_only=False)

chat = ChatTTS.Chat()
device=os.getenv('device','default')
chat.load(source="local",device=None if device=='default' else device, compile=True if os.getenv('compile','true').lower()!='false' else False)

#-------log----

log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

app = Flask(__name__, 
    static_folder=ROOT_DIR+'/static', 
    static_url_path='/static',
    template_folder=ROOT_DIR+'/templates')

root_log = logging.getLogger()
root_log.handlers = []
root_log.setLevel(logging.WARNING)
app.logger.setLevel(logging.WARNING) 

file_handler = RotatingFileHandler(LOGS_DIR+f'/{datetime.datetime.now().strftime("%Y%m%d")}.log', maxBytes=1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)
app.jinja_env.globals.update(enumerate=enumerate)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


@app.route('/')
def index():
    speakers=utils.get_speakers()
    return render_template(
        f"index{'' if is_cn else 'en'}.html",
        weburl=WEB_ADDRESS,
        speakers=speakers,
        version=VERSION
    )


# 根据文本返回tts结果，返回 filename=文件名 url=可下载地址
# 请求端根据需要自行选择使用哪个
# params:
#
# text:待合成文字
# prompt：
# voice：音色
# custom_voice：自定义音色值
# skip_refine: 1=跳过refine_text阶段，0=不跳过
# temperature
# top_p
# top_k
# speed
# text_seed
# refine_max_new_token
# infer_max_new_token
# wav

audio_queue=[]

@app.route('/tts', methods=['GET', 'POST'])
def tts():
    global audio_queue
    # 原始字符串
    text = request.args.get("text","").strip() or request.form.get("text","").strip()
    prompt = request.args.get("prompt","").strip() or request.form.get("prompt",'')

    # 默认值
    defaults = {
        "custom_voice": 0,
        "voice": "2222",
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 20,
        "skip_refine": 0,
        "speed":5,
        "text_seed":42,
        "refine_max_new_token": 384,
        "infer_max_new_token": 2048,
        "wav": 0,
        "is_stream":0
    }

    # 获取
    custom_voice = utils.get_parameter(request, "custom_voice", defaults["custom_voice"], int)
    voice = str(custom_voice) if custom_voice > 0 else utils.get_parameter(request, "voice", defaults["voice"], str)
    temperature = utils.get_parameter(request, "temperature", defaults["temperature"], float)
    top_p = utils.get_parameter(request, "top_p", defaults["top_p"], float)
    top_k = utils.get_parameter(request, "top_k", defaults["top_k"], int)
    skip_refine = utils.get_parameter(request, "skip_refine", defaults["skip_refine"], int)
    is_stream = utils.get_parameter(request, "is_stream", defaults["is_stream"], int)
    speed = utils.get_parameter(request, "speed", defaults["speed"], int)
    text_seed = utils.get_parameter(request, "text_seed", defaults["text_seed"], int)
    refine_max_new_token = utils.get_parameter(request, "refine_max_new_token", defaults["refine_max_new_token"], int)
    infer_max_new_token = utils.get_parameter(request, "infer_max_new_token", defaults["infer_max_new_token"], int)
    wav = utils.get_parameter(request, "wav", defaults["wav"], int)
        
        
    
    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})
    # 固定音色
    rand_spk=None
    # voice可能是 {voice}.csv or {voice}.pt or number
    seed_path=f'{SPEAKER_DIR}/{voice}'
    print(f'{voice=}')
    if voice.endswith('.csv') and os.path.exists(seed_path):
        rand_spk=utils.load_speaker(voice)
        print(f'当前使用音色 {seed_path=}')
    elif voice.endswith('.pt') and os.path.exists(seed_path):
        #如果.env中未指定设备，则使用 ChatTTS相同算法找设备，否则使用指定设备
        rand_spk=torch.load(seed_path, map_location=select_device(4096) if device=='default' else torch.device(device))
        print(f'当前使用音色 {seed_path=}')
    # 否则 判断是否存在 {voice}.csv
    elif os.path.exists(f'{SPEAKER_DIR}/{voice}.csv'):
        rand_spk=utils.load_speaker(voice)
        print(f'当前使用音色 {SPEAKER_DIR}/{voice}.csv')
    
    if rand_spk is None:    
        print(f'当前使用音色：根据seed={voice}获取随机音色')
        voice=int(voice) if re.match(r'^\d+$',voice) else 2222
        torch.manual_seed(voice)
        std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
        #rand_spk = chat.sample_random_speaker()
        rand_spk = torch.randn(768) * std + mean
        # 保存音色
        utils.save_speaker(voice,rand_spk)
        

    audio_files = []
    start_time = time.time()
    
    # 中英按语言分行
    text_list=[t.strip() for t in text.split("\n") if t.strip()]
    new_text=utils.split_text(text_list)
    if text_seed>0:
        torch.manual_seed(text_seed)
    
    
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        prompt=f"[speed_{speed}]",
        top_P=top_p,
        top_K=top_k,
        temperature=temperature,
        max_new_token=infer_max_new_token
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt=prompt,
        top_P=top_p,
        top_K=top_k,
        temperature=temperature,
        max_new_token=refine_max_new_token
    )
    
    wavs = chat.infer(
        new_text, 
        use_decoder=True,
        stream=True if is_stream==1 else False,
        skip_refine_text=skip_refine,
        do_text_normalization=False,
        do_homophone_replacement=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code
        
        )
    combined_wavdata=None
    end_time = time.time()
    inference_time = end_time - start_time
    inference_time_rounded = round(inference_time, 2)
    print(f"推理时长: {inference_time_rounded} 秒")

    wav_list = []
    for wavdata in wavs:
        # np.squeeze 把可能的二维数组 (1, N) 压扁成一维 (N,)
        # 如果它已经是 (N,) 一维数组，则保持不变
        flat_wav = np.squeeze(wavdata)
        wav_list.append(flat_wav)
        
    # 一次性合并所有音频片段
    if wav_list:
        combined_wavdata = np.concatenate(wav_list)
    else:
        combined_wavdata = np.array([])

    sample_rate = 24000  # Assuming 24kHz sample rate
    audio_duration = len(combined_wavdata) / sample_rate
    audio_duration_rounded = round(audio_duration, 2)
    print(f"音频时长: {audio_duration_rounded} 秒")
    
    
    filename = datetime.datetime.now().strftime('%H%M%S_')+f"use{inference_time_rounded}s-audio{audio_duration_rounded}s-seed{voice}-te{temperature}-tp{top_p}-tk{top_k}-textlen{len(text)}-{str(random())[2:7]}" + ".wav"
    sf.write(WAVS_DIR+'/'+filename, combined_wavdata, 24000)

    audio_files.append({
        "filename": WAVS_DIR + '/' + filename,
        "url": f"http://{request.host}/static/wavs/{filename}",
        "inference_time": inference_time_rounded,
        "audio_duration": audio_duration_rounded
    })
    result_dict={"code": 0, "msg": "ok", "audio_files": audio_files}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    # 兼容pyVideoTrans接口调用
    if len(audio_files)==1:
        result_dict["filename"]=audio_files[0]['filename']
        result_dict["url"]=audio_files[0]['url']

    if wav>0:
        return send_file(audio_files[0]['filename'], mimetype='audio/x-wav')
    else:
        return jsonify(result_dict)



@app.route('/clear_wavs', methods=['POST'])
def clear_wavs():
    dir_path = 'static/wavs'  # wav音频文件存储目录
    success, message = utils.ClearWav(dir_path)
    if success:
        return jsonify({"code": 0, "msg": message})
    else:
        return jsonify({"code": 1, "msg": message})

try:
    host = WEB_ADDRESS.split(':')
    print(f'Start:{WEB_ADDRESS}')
    threading.Thread(target=utils.openweb,args=(f'http://{WEB_ADDRESS}',)).start()
    serve(app,host=host[0], port=int(host[1]))
except Exception as e:
    print(e)

