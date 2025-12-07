import os
import re
import sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import io
import json
import torchaudio
import wave
import zipfile
from pathlib import Path

print("Starting...")
import shutil
import time

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import subprocess
import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv

load_dotenv()
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory,
    send_file,
    Response,
    stream_with_context,
)
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
from random import random
from modelscope import snapshot_download
import numpy as np
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils,VERSION
from uilib.tts_service import run_tts_generation
from uilib.audio_service import list_audio_files, delete_audio_file
from ChatTTS.utils import select_device
from uilib.utils import is_chinese_os, modelscope_status

merge_size = int(os.getenv("merge_size", 10))
env_lang = os.getenv("lang", "")
if env_lang == "zh":
    is_cn = True
elif env_lang == "en":
    is_cn = False
else:
    is_cn = is_chinese_os()

if not shutil.which("ffmpeg"):
    print("请先安装ffmpeg")
    time.sleep(60)
    exit()


chat = ChatTTS.Chat()
device_str = os.getenv("device", "default")

if device_str in ["default", "mps"]:
    device = select_device(min_memory=2047, experimental=True if device_str == "mps" else False)
elif device_str == "cuda":
    device = select_device(min_memory=2047)
elif device_str == "cpu":
    device = torch.device("cpu")


chat.load(
    source="local" if not os.path.exists(MODEL_DIR + "/DVAE_full.pt") else "custom",
    custom_path=ROOT_DIR,
    device=device,
    compile=True if os.getenv("compile", "true").lower() != "false" else False,
)


# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger("werkzeug")
log.handlers[:] = []
log.setLevel(logging.WARNING)

app = Flask(
    __name__,
    static_folder=ROOT_DIR + "/static",
    static_url_path="/static",
    template_folder=ROOT_DIR + "/templates",
)

root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.WARNING)
app.logger.setLevel(logging.WARNING)
# 创建 RotatingFileHandler 对象，设置写入的文件路径和大小限制
file_handler = RotatingFileHandler(
    LOGS_DIR + f'/{datetime.datetime.now().strftime("%Y%m%d")}.log',
    maxBytes=1024 * 1024,
    backupCount=5,
)
# 创建日志的格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# 设置文件处理器的级别和格式
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
# 将文件处理器添加到日志记录器中
app.logger.addHandler(file_handler)
app.jinja_env.globals.update(enumerate=enumerate)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)


@app.route("/")
def index():
    speakers = utils.get_speakers()
    lang = request.args.get("lang", "").strip().lower()
    if lang == "zh":
        tpl = "index.html"
    elif lang == "en":
        tpl = "indexen.html"
    else:
        tpl = f"index{'' if is_cn else 'en'}.html"
    return render_template(
        tpl,
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

audio_queue = []


@app.route("/tts", methods=["GET", "POST"])
def tts():
    global audio_queue
    # 原始字符串
    text = request.args.get("text", "").strip() or request.form.get("text", "").strip()
    prompt = request.args.get("prompt", "").strip() or request.form.get("prompt", "")

    # 默认值
    defaults = {
        "custom_voice": 0,
        "voice": "2222",
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 20,
        "skip_refine": 0,
        "speed": 5,
        "text_seed": 42,
        "refine_max_new_token": 384,
        "infer_max_new_token": 2048,
        "wav": 0,
        "is_stream": 0,
        # 是否返回分片音频，1=返回所有分片+合并文件，0=仅返回合并文件
        "split": 0,
        # 文本分片长度，0=自动（保持默认行为）
        "segment_len": 0,
        # 目标分片时长（秒），0=不按时长，仅按字符数
        "segment_seconds": 0.0,
    }

    # 获取
    custom_voice = utils.get_parameter(request, "custom_voice", defaults["custom_voice"], int)
    voice = (
        str(custom_voice)
        if custom_voice > 0
        else utils.get_parameter(request, "voice", defaults["voice"], str)
    )
    temperature = utils.get_parameter(request, "temperature", defaults["temperature"], float)
    top_p = utils.get_parameter(request, "top_p", defaults["top_p"], float)
    top_k = utils.get_parameter(request, "top_k", defaults["top_k"], int)
    skip_refine = utils.get_parameter(request, "skip_refine", defaults["skip_refine"], int)
    is_stream = utils.get_parameter(request, "is_stream", defaults["is_stream"], int)
    speed = utils.get_parameter(request, "speed", defaults["speed"], int)
    text_seed = utils.get_parameter(request, "text_seed", defaults["text_seed"], int)
    refine_max_new_token = utils.get_parameter(
        request, "refine_max_new_token", defaults["refine_max_new_token"], int
    )
    infer_max_new_token = utils.get_parameter(
        request, "infer_max_new_token", defaults["infer_max_new_token"], int
    )
    wav = utils.get_parameter(request, "wav", defaults["wav"], int)
    split_mode = utils.get_parameter(request, "split", defaults["split"], int)
    segment_len = utils.get_parameter(request, "segment_len", defaults["segment_len"], int)
    segment_seconds = utils.get_parameter(request, "segment_seconds", defaults["segment_seconds"], float)
        
        
    
    # 如果指定了分片目标时长，则将其近似映射为字符长度
    if segment_seconds and segment_seconds > 0:
        # 简单近似：每秒约 12 个字符，可根据需要在 .env 中调整为环境变量
        approx_chars_per_second = 12
        try:
            approx_chars_per_second = int(os.getenv('CHARS_PER_SECOND', approx_chars_per_second))
        except Exception:
            pass
        segment_len = max(1, int(segment_seconds * approx_chars_per_second))

    app.logger.info(f"[tts]{text=}\n{voice=},{skip_refine=}, {segment_len=}, {segment_seconds=}\n")
    if not text:
        return jsonify({"code": 1, "msg": "text params lost"})

    try:
        audio_files = run_tts_generation(
            chat=chat,
            text=text,
            prompt=prompt,
            voice=voice,
            custom_voice=custom_voice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            skip_refine=skip_refine,
            is_stream=is_stream,
            speed=speed,
            text_seed=text_seed,
            refine_max_new_token=refine_max_new_token,
            infer_max_new_token=infer_max_new_token,
            split_mode=split_mode,
            segment_len=segment_len,
            merge_size=merge_size,
            device=device,
        )
    except RuntimeError as e:
        return jsonify({"code": 1, "msg": str(e)})

    result_dict={"code": 0, "msg": "ok", "audio_files": audio_files}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    # 兼容 pyVideoTrans 接口调用：当只有一个音频时，补充 filename/url 字段
    if len(audio_files) == 1:
        result_dict["filename"] = audio_files[0]["filename"]
        result_dict["url"] = audio_files[0]["url"]

    # 如果请求直接返回 wav 文件，则优先返回合并后的音频（如存在）
    if wav > 0:
        target = audio_files[0]
        for it in audio_files:
            if isinstance(it, dict) and it.get("is_merged") == 1:
                target = it
                break
        return send_file(target["filename"], mimetype="audio/x-wav")

    return jsonify(result_dict)


@app.route("/clear_wavs", methods=["POST"])
def clear_wavs():
    dir_path = "static/wavs"  # wav音频文件存储目录
    success, message = utils.ClearWav(dir_path)
    if success:
        return jsonify({"code": 0, "msg": message})
    else:
        return jsonify({"code": 1, "msg": message})


@app.route('/list_wavs', methods=['GET'])
def list_wavs():
    """列出当前 static/wavs 下的所有 wav 文件。"""
    try:
        audio_files = list_audio_files()
        return jsonify({"code": 0, "msg": "ok", "audio_files": audio_files})
    except Exception as e:
        return jsonify({"code": 1, "msg": str(e), "audio_files": []})


@app.route('/delete_wav', methods=['POST'])
def delete_wav():
    """删除单个 wav 文件。只允许删除 static/wavs 下的文件。"""
    filename = request.form.get("filename", "").strip()
    if not filename and request.is_json:
        data = request.get_json(silent=True) or {}
        filename = str(data.get("filename", "")).strip()

    try:
        delete_audio_file(filename)
        return jsonify({"code": 0, "msg": "deleted"})
    except (ValueError, FileNotFoundError) as e:
        # 与原有返回保持一致：code=1, msg 为错误原因
        return jsonify({"code": 1, "msg": str(e)})
    except Exception as e:
        return jsonify({"code": 1, "msg": str(e)})


@app.route('/delete_wavs_batch', methods=['POST'])
def delete_wavs_batch():
    """Delete multiple wav files in one request.

    Expects either:
    - form field 'filenames' as comma/newline separated string, or
    - JSON body {"filenames": ["file1.wav", "file2.wav", ...]}
    """

    filenames = []

    raw = request.form.get("filenames", "").strip()
    if raw:
        for name in re.split(r"[\n,]", raw):
            name = name.strip()
            if name:
                filenames.append(name)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        arr = data.get("filenames", [])
        if isinstance(arr, list):
            for name in arr:
                name = str(name or "").strip()
                if name:
                    filenames.append(name)

    if not filenames:
        return jsonify({"code": 1, "msg": "filenames is required"})

    errors = []
    deleted = 0
    for name in filenames:
        try:
            delete_audio_file(name)
            deleted += 1
        except (ValueError, FileNotFoundError) as e:
            errors.append(f"{name}: {e}")
        except Exception as e:
            errors.append(f"{name}: {e}")

    if errors:
        return jsonify({"code": 1, "msg": "; ".join(errors), "deleted": deleted})

    return jsonify({"code": 0, "msg": "deleted", "deleted": deleted})


@app.route('/download_wavs_batch', methods=['POST'])
def download_wavs_batch():
    """打包下载多个 wav 文件为一个 zip。

    接收参数格式与 /delete_wavs_batch 相同：
    - form 字段 'filenames' 为逗号/换行分隔的字符串，或
    - JSON {"filenames": ["file1.wav", "file2.wav", ...]}
    """

    filenames = []

    raw = request.form.get("filenames", "").strip()
    if raw:
        for name in re.split(r"[\n,]", raw):
            name = name.strip()
            if name:
                filenames.append(name)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        arr = data.get("filenames", [])
        if isinstance(arr, list):
            for name in arr:
                name = str(name or "").strip()
                if name:
                    filenames.append(name)

    if not filenames:
        return jsonify({"code": 1, "msg": "filenames is required"}), 400

    buf = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            # 安全检查：仅允许纯文件名，禁止路径穿越
            if "/" in name or "\\" in name:
                continue
            file_path = os.path.join(WAVS_DIR, name)
            if not os.path.isfile(file_path):
                continue
            zf.write(file_path, arcname=name)
            added += 1

    if not added:
        return jsonify({"code": 1, "msg": "no valid files to download"}), 400

    buf.seek(0)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"chattts_batch_{added}_{ts}.zip"
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


try:
    host = WEB_ADDRESS.split(':')
    print(f'Start:{WEB_ADDRESS}')
    threading.Thread(target=utils.openweb,args=(f'http://{WEB_ADDRESS}',)).start()
    serve(app,host=host[0], port=int(host[1]))
except Exception as e:
    print(e)
