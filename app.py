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
from flask import Flask, request, render_template, jsonify,  send_from_directory,send_file,Response, stream_with_context
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
from random import random
from modelscope import snapshot_download
import numpy as np
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils,VERSION
from ChatTTS.utils import select_device
from uilib.utils import is_chinese_os,modelscope_status
merge_size=int(os.getenv('merge_size',10))
env_lang=os.getenv('lang','')
if env_lang=='zh':
    is_cn= True
elif env_lang=='en':
    is_cn=False
else:
    is_cn=is_chinese_os()
    
if not shutil.which("ffmpeg"):
    print('请先安装ffmpeg')
    time.sleep(60)
    exit()    


chat = ChatTTS.Chat()
device_str=os.getenv('device','default')

if device_str in ['default','mps']:
    device=select_device(min_memory=2047,experimental=True if device_str=='mps' else False)
elif device_str =='cuda':
    device=select_device(min_memory=2047)
elif device_str == 'cpu':
    device = torch.device("cpu")


chat.load(source="local" if not os.path.exists(MODEL_DIR+"/DVAE_full.pt") else 'custom',custom_path=ROOT_DIR, device=device,compile=True if os.getenv('compile','true').lower()!='false' else False)


# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

app = Flask(__name__, 
    static_folder=ROOT_DIR+'/static', 
    static_url_path='/static',
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
    voice=voice.replace('.csv','.pt')
    seed_path=f'{SPEAKER_DIR}/{voice}'
    print(f'{voice=}')
    #if voice.endswith('.csv') and os.path.exists(seed_path):
    #    rand_spk=utils.load_speaker(voice)
    #    print(f'当前使用音色 {seed_path=}')
    #el
    
    if voice.endswith('.pt') and os.path.exists(seed_path):
        #如果.env中未指定设备，则使用 ChatTTS相同算法找设备，否则使用指定设备
        rand_spk=torch.load(seed_path, map_location=device)
        print(f'当前使用音色 {seed_path=}')
    # 否则 判断是否存在 {voice}.csv
    #elif os.path.exists(f'{SPEAKER_DIR}/{voice}.csv'):
    #    rand_spk=utils.load_speaker(voice)
    #    print(f'当前使用音色 {SPEAKER_DIR}/{voice}.csv')
    
    if rand_spk is None:    
        print(f'当前使用音色：根据seed={voice}获取随机音色')
        voice_int=re.findall(r'^(\d+)',voice)
        if len(voice_int)>0:
            voice=int(voice_int[0])
        else:
            voice=2222
        torch.manual_seed(voice)
        #std, mean = chat.sample_random_speaker
        rand_spk = chat.sample_random_speaker()
        #rand_spk = torch.randn(768) * std + mean
        # 保存音色
        torch.save(rand_spk,f"{SPEAKER_DIR}/{voice}.pt")
        #utils.save_speaker(voice,rand_spk)
        

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
    print(f'{prompt=}')
    # 将少于30个字符的行同其他行拼接
    retext=[]
    short_text=""
    for it in new_text:
        if len(it)<30:
            short_text+=f"{it} [uv_break] "
            if len(short_text)>30:
                retext.append(short_text)
                short_text=""
        else:
            retext.append(short_text+it)
            short_text=""
    if len(short_text)>30 or len(retext)<1:
        retext.append(short_text)
    elif short_text:
        retext[-1]+=f" [uv_break] {short_text}"
        
    new_text=retext
    
    new_text_list=[new_text[i:i+merge_size] for i in range(0,len(new_text),merge_size)]
    filename_list=[]

    audio_time=0
    inter_time=0

    for i,te in enumerate(new_text_list):
        print(f'{te=}')
        wavs = chat.infer(
            te, 
            #use_decoder=False,
            stream=True if is_stream==1 else False,
            skip_refine_text=skip_refine,
            do_text_normalization=False,
            do_homophone_replacement=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code
            
            )


        end_time = time.time()
        inference_time = end_time - start_time
        inference_time_rounded = round(inference_time, 2)
        inter_time+=inference_time_rounded
        print(f"推理时长: {inference_time_rounded} 秒")

       
        
        for j,w in enumerate(wavs):
            filename = datetime.datetime.now().strftime('%H%M%S_')+f"use{inference_time_rounded}s-seed{voice}-te{temperature}-tp{top_p}-tk{top_k}-textlen{len(text)}-{str(random())[2:7]}" + f"-{i}-{j}.wav"
            filename_list.append(filename)
            torchaudio.save(WAVS_DIR+'/'+filename, torch.from_numpy(w).unsqueeze(0), 24000)
        
    txt_tmp="\n".join([f"file '{WAVS_DIR}/{it}'" for it in filename_list])
    txt_name=f'{time.time()}.txt'
    with open(f'{WAVS_DIR}/{txt_name}','w',encoding='utf-8') as f:
        f.write(txt_tmp)
    outname=datetime.datetime.now().strftime('%H%M%S_')+f"use{inter_time}s-audio{audio_time}s-seed{voice}-te{temperature}-tp{top_p}-tk{top_k}-textlen{len(text)}-{str(random())[2:7]}" + "-merge.wav"
    try:
        subprocess.run(["ffmpeg","-hide_banner", "-ignore_unknown","-y","-f","concat","-safe","0","-i",f'{WAVS_DIR}/{txt_name}',"-c:a","copy",WAVS_DIR + '/' + outname],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   encoding="utf-8",
                   check=True,
                   text=True,
                   creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        return jsonify({"code":1,"msg":str(e)})

    

    audio_files.append({
        "filename": WAVS_DIR + '/' + outname,
        "url": f"http://{request.host}/static/wavs/{outname}",
        "inference_time": round(inter_time,2),
        "audio_duration": -1
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

