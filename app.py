import os
import re
import sys
from pathlib import Path
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify,  send_from_directory
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
load_dotenv()
from random import random
from modelscope import snapshot_download
import numpy as np
import time
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils,VERSION
from flask import Flask, send_file
import io
import wave
CHATTTS_DIR= MODEL_DIR+'/pzc163/chatTTS'
# 默认从 modelscope 下载模型,如果想从huggingface下载模型，请将以下代码注释掉
# 如果已存在则不再下载和检测更新，便于离线内网使用
if not os.path.exists(CHATTTS_DIR+"/config/path.yaml"):
    snapshot_download('pzc163/chatTTS',cache_dir=MODEL_DIR)
chat = ChatTTS.Chat()
chat.load_models(source="local",local_path=CHATTTS_DIR, compile=True if os.getenv('compile','true').lower()!='false' else False)

# 如果希望从 huggingface.co下载模型，将以下注释删掉。将上方3行内容注释掉
# 如果已存在则不再下载和检测更新，便于离线内网使用
#CHATTTS_DIR=MODEL_DIR+'/models--2Noise--ChatTTS'
#if not os.path.exists(CHATTTS_DIR):
    #import huggingface_hub
    #os.environ['HF_HUB_CACHE']=MODEL_DIR
    #os.environ['HF_ASSETS_CACHE']=MODEL_DIR
    #huggingface_hub.snapshot_download(cache_dir=MODEL_DIR,repo_id="2Noise/ChatTTS", allow_patterns=["*.pt", "*.yaml"])
    # chat = ChatTTS.Chat()
    # chat.load_models(source="local",local_path=CHATTTS_DIR, compile=True if os.getenv('compile','true').lower()!='false' else False)


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
    return render_template("index.html",weburl=WEB_ADDRESS,version=VERSION)


# 根据文本返回tts结果，返回 filename=文件名 url=可下载地址
# 请求端根据需要自行选择使用哪个
# params:
#
# text:待合成文字
# voice：音色
# custom_voice：自定义音色值
# skip_refine: 1=跳过refine_text阶段，0=不跳过
# is_split: 1=启用中英分词，同时将数字转为对应语言发音，0=不启用
# temperature
# top_p
# top_k
# prompt：
@app.route('/tts', methods=['GET', 'POST'])
def tts():
    # 原始字符串
    text = request.args.get("text","").strip() or request.form.get("text","").strip()
    prompt = request.form.get("prompt",'')
    try:
        custom_voice=int(request.form.get("custom_voice",0))
        voice =  custom_voice if custom_voice>0  else int(request.form.get("voice",2222))
    except Exception:
        voice=2222
    print(f'{voice=},{custom_voice=}')
    temperature = float(request.form.get("temperature",0.3))
    top_p = float(request.form.get("top_p",0.7))
    top_k = int(request.form.get("top_k",20))
    skip_refine=0
    is_split=0
    speed=5
    refine_max_new_token=384
    infer_max_new_token=2048
    text_seed=42
    try:
        skip_refine = int(request.form.get("skip_refine",0))
        is_split = int(request.form.get("is_split",0))
    except Exception as e:
        print(e)
    try:
        text_seed = int(request.form.get("text_seed",42))
    except Exception as e:
        print(e)
    try:
        speed = int(request.form.get("speed",5))
    except Exception as e:
        print(e)
    try:
        refine_max_new_token=int(request.form.get("refine_max_new_token",384))
        infer_max_new_token=int(request.form.get("infer_max_new_token",2048))
    except Exception as e:
        print(e)
    
    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})
    # 固定音色
    rand_spk=utils.load_speaker(voice)
    if rand_spk is None:    
        print(f'根据seed={voice}获取随机音色')
        torch.manual_seed(voice)
        std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
        #rand_spk = chat.sample_random_speaker()        
        rand_spk = torch.randn(768) * std + mean
        # 保存音色
        utils.save_speaker(voice,rand_spk)
    else:
        print(f'固定音色 seed={voice}')

    audio_files = []
    

    start_time = time.time()
    
    # 中英按语言分行
    text_list=[t.strip() for t in text.split("\n") if t.strip()]
    new_text=text_list if is_split==0 else utils.split_text(text_list)
    if text_seed>0:
        torch.manual_seed(text_seed)
    print(f'{text_seed=}')
    print(f'[speed_{speed}]')
    wavs = chat.infer(new_text, use_decoder=True, skip_refine_text=True if int(skip_refine)==1 else False,params_infer_code={
        'spk_emb': rand_spk,
        'prompt':f'[speed_{speed}]',
        'temperature':temperature,
        'top_P':top_p,
        'top_K':top_k,
        'max_new_token':infer_max_new_token
    }, params_refine_text= {'prompt': prompt,'max_new_token':refine_max_new_token},do_text_normalization=False)

    end_time = time.time()
    inference_time = end_time - start_time
    inference_time_rounded = round(inference_time, 2)
    print(f"推理时长: {inference_time_rounded} 秒")

    # 初始化一个空的numpy数组用于之后的合并
    combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # 确保dtype与你的wav数据类型匹配

    for wavdata in wavs:
        combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))

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
    # 兼容pyVideoTrans接口调用
    if len(audio_files)==1:
        result_dict["filename"]=audio_files[0]['filename']
        result_dict["url"]=audio_files[0]['url']

    return jsonify(result_dict)


@app.route('/tts1', methods=['GET', 'POST'])
def tts1():
    # 原始字符串
    text = request.args.get("text","").strip() or request.form.get("text","").strip()
    prompt = request.form.get("prompt",'')
    try:  
# 尝试从 request.args 获取参数  
        custom_voice_str = request.args.get("custom_voice", "")  
  
        # 如果 request.args 中的参数不为空字符串，则尝试转换为整数  
        if custom_voice_str.strip():  
            try:  
                custom_voice = int(custom_voice_str)  
            except ValueError:  
                # 如果转换失败，则使用 request.form 中的默认值  
                custom_voice = int(request.form.get("custom_voice", 0))  
        else:  
            # 如果 request.args 中没有参数或为空字符串，则从 request.form 获取  
            custom_voice = int(request.form.get("custom_voice", 0))  
  
        # 现在 custom_voice 已经设置好了
        if custom_voice > 0:  
            voice = custom_voice  
        else:  
            try:  
# 尝试从 request.args 获取参数  
                voice_str = request.args.get("voice", "")  
  
                # 如果从 request.args 获取到了非空字符串，则尝试将其转换为整数  
                if voice_str.strip():  
                    voice = int(voice_str)  
                else:  
                    # 如果没有从 request.args 获取到参数或为空字符串，则从 request.form 获取  
                    voice = int(request.form.get("voice", 2222))  
  
                # 现在 voice 已经设置好了
            except ValueError:  
                voice = 2222  
    except ValueError:  
    # 如果 "custom" 也不是一个可以转换为整数的值，但在这个情况下我们已经有了一个默认值 0  
        voice = 2222  
  
    print(f'{voice=},{custom_voice=}')
	
	
# 尝试从 request.args 获取参数  
    temperature_str = request.args.get("temperature", "")  
    top_p_str = request.args.get("top_p", "")  
    top_k_str = request.args.get("top_k", "")  
  
    # 如果 request.args 中的参数为空字符串，则尝试从 request.form 获取  
    if not temperature_str.strip():  
        temperature_str = str(request.form.get("temperature", 0.3))  
    if not top_p_str.strip():  
        top_p_str = str(request.form.get("top_p", 0.7))  
    if not top_k_str.strip():  
        top_k_str = str(request.form.get("top_k", 20))  
  
    # 将字符串转换为相应的数据类型，同时处理可能的异常  
    try:  
        temperature = float(temperature_str)  
        top_p = float(top_p_str)  
        top_k = int(top_k_str)  
    except ValueError:  
        # 如果转换失败，则可以使用默认值或其他错误处理逻辑  
        temperature = 0.3  
        top_p = 0.7  
        top_k = 20  
        speed = 5  
    # 现在 temperature, top_p, top_k 已经根据优先级被设置好了
    skip_refine=0
    is_split=0
    speed=5
    refine_max_new_token=384
    infer_max_new_token=2048
    text_seed=42
    try:
# 尝试从 request.args 获取参数  
        skip_refine_str = request.args.get("skip_refine", "")  
        is_split_str = request.args.get("is_split", "")  
  
        # 如果 request.args 中的参数为空字符串，则尝试从 request.form 获取  
        if not skip_refine_str.strip():  
            skip_refine_str = str(request.form.get("skip_refine", 0))  
        if not is_split_str.strip():  
            is_split_str = str(request.form.get("is_split", 0))  

        # 将字符串转换为相应的数据类型，同时处理可能的异常  
        try:  
            skip_refine = int(skip_refine_str)  
            is_split = int(is_split_str)  
            speed = int(speed_str)
        except ValueError:  
            # 如果转换失败，则使用默认值  
            skip_refine = 0  
            is_split = 0  
  
        # 现在 skip_refine 和 is_split 已经根据优先级被设置好了
    except Exception as e:
        print(e)

    try:
# 尝试从 request.args 获取参数  
        speed_str = request.args.get("speed", "")  
        text_seed_str = request.args.get("text_seed", "")  
  
        # 如果 request.args 中的参数为空字符串，则尝试从 request.form 获取  
        if not speed_str.strip():  
            speed_str = str(request.form.get("speed", 5))  
        if not text_seed_str.strip():  
            text_seed_str = str(request.form.get("text_seed", 42))  

        # 将字符串转换为相应的数据类型，同时处理可能的异常  
        try:  
            speed = int(speed_str)  
            text_seed = int(text_seed_str)  
            speed = int(speed_str)
        except ValueError:  
            # 如果转换失败，则使用默认值  
            speed = 5  
            text_seed = 42  
  
        # 现在 speed 和 text_seed 已经根据优先级被设置好了
    except Exception as e:
        print(e)

    try:
# 尝试从 request.args 获取参数  
        refine_max_new_token_str = request.args.get("refine_max_new_token", "")  
        infer_max_new_token_str = request.args.get("infer_max_new_token", "")  
  
        # 如果 request.args 中的参数为空字符串，则尝试从 request.form 获取  
        if not refine_max_new_token_str.strip():  
            refine_max_new_token_str = str(request.form.get("refine_max_new_token", 384))  
        if not infer_max_new_token_str.strip():  
            infer_max_new_token_str = str(request.form.get("infer_max_new_token", 2048))  
  
        # 将字符串转换为相应的数据类型，同时处理可能的异常  
        try:  
            refine_max_new_token = int(refine_max_new_token_str)  
            infer_max_new_token = int(infer_max_new_token_str)  
        except ValueError:  
            # 如果转换失败，则使用默认值  
            refine_max_new_token = 384  
            infer_max_new_token = 2048  
  
        # 现在 refine_max_new_token 和 infer_max_new_token 已经根据优先级被设置好了
    except Exception as e:
        print(e)
    
    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})
    # 固定音色
    rand_spk=utils.load_speaker(voice)
    if rand_spk is None:    
        print(f'根据seed={voice}获取随机音色')
        torch.manual_seed(voice)
        std, mean = torch.load(f'{CHATTTS_DIR}/asset/spk_stat.pt').chunk(2)
        #rand_spk = chat.sample_random_speaker()        
        rand_spk = torch.randn(768) * std + mean
        # 保存音色
        utils.save_speaker(voice,rand_spk)
    else:
        print(f'固定音色 seed={voice}')

    audio_files = []
    

    start_time = time.time()
    
    # 中英按语言分行
    text_list=[t.strip() for t in text.split("\n") if t.strip()]
    new_text=text_list if is_split==0 else utils.split_text(text_list)

    if text_seed>0:
        torch.manual_seed(text_seed)
    print(f'{text_seed=}')
    print(f'[speed_{speed}]')
    wavs = chat.infer(new_text, use_decoder=True, skip_refine_text=True if int(skip_refine)==1 else False,params_infer_code={
        'spk_emb': rand_spk,
        'prompt':f'[speed_{speed}]',
        'temperature':temperature,
        'top_P':top_p,
        'top_K':top_k,
        'max_new_token':infer_max_new_token
    }, params_refine_text= {'prompt': prompt,'max_new_token':refine_max_new_token},do_text_normalization=False)

    end_time = time.time()
    inference_time = end_time - start_time
    inference_time_rounded = round(inference_time, 2)
    print(f"推理时长: {inference_time_rounded} 秒")

    # 初始化一个空的numpy数组用于之后的合并
    combined_wavdata = np.array([], dtype=wavs[0][0].dtype)  # 确保dtype与你的wav数据类型匹配

    for wavdata in wavs:
        combined_wavdata = np.concatenate((combined_wavdata, wavdata[0]))

    sample_rate = 24000  # Assuming 24kHz sample rate
    audio_duration = len(combined_wavdata) / sample_rate
    audio_duration_rounded = round(audio_duration, 2)
    print(f"音频时长: {audio_duration_rounded} 秒")
    
    
    filename = datetime.datetime.now().strftime('%H%M%S_') + ".wav"
    sf.write(WAVS_DIR+'/'+filename, combined_wavdata, 24000)

    audio_files.append({
         "filename": WAVS_DIR + '/' + filename,
         "url": f"http://{request.host}/static/wavs/{filename}",
         "inference_time": inference_time_rounded,
         "audio_duration": audio_duration_rounded
     })
    #result_dict={"code": 0, "msg": "ok", "audio_files": audio_files}
    #兼容pyVideoTrans接口调用
    #if len(audio_files)==1:
    #    result_dict["filename"]=audio_files[0]['filename']
    #    result_dict["url"]=audio_files[0]['url']
    return send_file(audio_files[0]['filename'], mimetype='audio/x-wav')



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
    print(f'启动:{WEB_ADDRESS}')
    threading.Thread(target=utils.openweb,args=(f'http://{WEB_ADDRESS}',)).start()
    serve(app,host=host[0], port=int(host[1]))
except Exception as e:
    print(e)

