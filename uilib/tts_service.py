import os
import sys
import re
import time
import datetime
import subprocess
from random import random

import torch
import torchaudio
import soundfile as sf
import ChatTTS

from uilib.cfg import SPEAKER_DIR, WAVS_DIR, WEB_ADDRESS
from uilib import utils


def _build_text_slug(text: str, max_len: int = 40) -> str:
    """Build a short, human-readable slug from text, ignoring control tokens."""

    # Normalize common non-breaking spaces and collapse whitespace
    txt = text.replace("\u00a0", " ")
    txt = " ".join(txt.split())

    # Drop ChatTTS control tokens like [uv_break], [laugh], [oral_2], etc.
    txt = re.sub(
        r"\[(?:uv_break|laugh|lbreak|break|oral_\d+|laugh_\d+|break_\d+)\]",
        " ",
        txt,
        flags=re.I,
    )
    txt = txt.strip()
    if not txt:
        return "audio"

    # Truncate early to avoid overly long filenames
    txt = txt[:max_len]

    # Replace whitespace with underscores
    slug = re.sub(r"\s+", "_", txt)
    # Remove characters invalid in filenames on common OSes (including '.', ',' and '，')
    slug = re.sub(r"[\\/:*?\"<>|\.,，]+", "_", slug)
    # Collapse duplicate underscores and trim
    slug = re.sub(r"_+", "_", slug).strip("_")

    return slug or "audio"


def _select_speaker(chat, voice: str, custom_voice: int, device):
    """Choose or create a speaker embedding based on voice/custom_voice.

    Returns (rand_spk, voice_for_filename).
    """

    rand_spk = None
    voice_str = str(custom_voice) if custom_voice and custom_voice > 0 else str(voice)
    voice_str = voice_str.replace(".csv", ".pt")
    seed_path = f"{SPEAKER_DIR}/{voice_str}"
    print(f"{voice_str=}")

    # 优先使用已有 pt
    if voice_str.endswith(".pt") and os.path.exists(seed_path):
        rand_spk = torch.load(seed_path, map_location=device)
        print(f"当前使用音色 {seed_path=}")

    voice_for_filename = voice_str
    if rand_spk is None:
        # 如果不存在 pt，则根据 seed 生成随机音色并保存
        print(f"当前使用音色：根据seed={voice_str}获取随机音色")
        voice_int = re.findall(r"^(\d+)", voice_str)
        if len(voice_int) > 0:
            used_voice = int(voice_int[0])
        else:
            used_voice = 2222
        torch.manual_seed(used_voice)
        # std, mean = chat.sample_random_speaker
        rand_spk = chat.sample_random_speaker()
        # rand_spk = torch.randn(768) * std + mean
        # 保存音色
        torch.save(rand_spk, f"{SPEAKER_DIR}/{used_voice}.pt")
        voice_for_filename = used_voice

    return rand_spk, voice_for_filename


def _split_and_group_text(text: str, segment_len: int, text_seed: int, merge_size: int):
    """Split raw text into grouped segments ready for inference."""

    # 中英按语言分行
    text_list = [t.strip() for t in text.split("\n") if t.strip()]
    new_text = utils.split_text(text_list, segment_len if segment_len and segment_len > 0 else None)
    if text_seed > 0:
        torch.manual_seed(text_seed)

    # 将少于 30 个字符的行同其他行拼接
    retext = []
    short_text = ""
    for it in new_text:
        if len(it) < 30:
            short_text += f"{it} [uv_break] "
            if len(short_text) > 30:
                retext.append(short_text)
                short_text = ""
        else:
            retext.append(short_text + it)
            short_text = ""
    if len(short_text) > 30 or len(retext) < 1:
        retext.append(short_text)
    elif short_text:
        retext[-1] += f" [uv_break] {short_text}"

    grouped = [retext[i : i + merge_size] for i in range(0, len(retext), merge_size)]
    return grouped


def _build_infer_params(rand_spk, speed, top_p, top_k, temperature, infer_max_new_token, prompt, refine_max_new_token):
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        prompt=f"[speed_{speed}]",
        top_P=top_p,
        top_K=top_k,
        temperature=temperature,
        max_new_token=infer_max_new_token,
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt=prompt,
        top_P=top_p,
        top_K=top_k,
        temperature=temperature,
        max_new_token=refine_max_new_token,
    )
    return params_infer_code, params_refine_text


def _infer_and_save_segments(
    chat,
    grouped_text,
    rand_spk,
    params_infer_code,
    params_refine_text,
    text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    voice_for_filename,
    start_time: float,
    skip_refine: int,
    is_stream: int,
):
    """Run inference on grouped_text, save each segment, and accumulate time stats."""

    filename_list = []
    audio_time = 0
    inter_time = 0

    for i, te in enumerate(grouped_text):
        print(f"{te=}")
        wavs = chat.infer(
            te,
            # use_decoder=False,
            stream=True if is_stream == 1 else False,
            skip_refine_text=skip_refine,
            # 我们在 uilib.utils 中已经完成了中英文归一化和数字处理, 继续关闭 ChatTTS 的文本归一化,
            # 但重新打开同音字替换功能, 并在 ChatTTS.Normalizer 内部对控制符做占位保护, 保证
            # [uv_break] / [laugh] / [break_6] / [oral_2] 等控制符不会被修改或删除, 同时恢复老版本
            # do_homophone_replacement=True 带来的整体语气/发音表现。
            do_text_normalization=False,
            do_homophone_replacement=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )

        end_time = time.time()
        inference_time = end_time - start_time
        inference_time_rounded = round(inference_time, 2)
        inter_time += inference_time_rounded
        print(f"推理时长: {inference_time_rounded} 秒")

        for j, w in enumerate(wavs):
            # Build slug from the corresponding text element for this wav
            if isinstance(te, (list, tuple)) and j < len(te):
                seg_text = str(te[j])
            else:
                seg_text = str(te)

            slug = _build_text_slug(seg_text)
            time_prefix = datetime.datetime.now().strftime("%H%M%S_")
            filename = (
                f"{time_prefix}{slug}_"
                + f"use{inference_time_rounded}s-seed{voice_for_filename}-te{temperature}-tp{top_p}-tk{top_k}-textlen{len(text)}-{str(random())[2:7]}"
                + f"-{i}-{j}.wav"
            )
            filename_list.append(filename)
            torchaudio.save(os.path.join(WAVS_DIR, filename), torch.from_numpy(w).unsqueeze(0), 24000)

    return filename_list, inter_time, audio_time


def _merge_segments(filename_list, inter_time, text, voice_for_filename, temperature, top_p, top_k, audio_time):
    """Merge segment wavs into a single file using ffmpeg and return path and duration."""

    txt_tmp = "\n".join([f"file '{os.path.join(WAVS_DIR, it)}'" for it in filename_list])
    txt_name = f"{time.time()}.txt"
    txt_path = os.path.join(WAVS_DIR, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_tmp)

    slug_base = _build_text_slug(text)
    slug = f"merged_{slug_base}"
    outname = (
        datetime.datetime.now().strftime("%H%M%S_")
        + f"{slug}_"
        + f"use{inter_time}s-audio{audio_time}s-seed{voice_for_filename}-te{temperature}-tp{top_p}-tk{top_k}-textlen{len(text)}-{str(random())[2:7]}"
        + "-merge.wav"
    )
    out_path = os.path.join(WAVS_DIR, outname)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-ignore_unknown",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                txt_path,
                "-c:a",
                "copy",
                out_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=True,
            text=True,
            creationflags=0 if sys.platform != "win32" else subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        # 交给上层处理
        raise RuntimeError(str(e))

    try:
        # 使用 soundfile 计算合并音频时长
        audio_info = sf.info(out_path)
        audio_duration = round(audio_info.duration, 2)
    except Exception as e:
        print(f"计算音频时长失败: {e}")
        audio_duration = -1

    return outname, out_path, audio_duration


def _build_full_url(relative_url: str) -> str:
    """Build absolute URL for audio files while keeping relative_url for UI.

    - Prefer current Flask request.host when running under HTTP.
    - Fallback to WEB_ADDRESS env (host:port) if no request context.
    - If both are unavailable, return the relative_url itself.
    """

    base = None

    # Try to use current HTTP request host if Flask request context exists
    try:
        from flask import request  # type: ignore

        host = getattr(request, "host", None)
    except Exception:
        host = None

    if host:
        base = host
    elif WEB_ADDRESS:
        base = WEB_ADDRESS

    if base:
        return f"http://{base}{relative_url}"
    return relative_url


def _build_audio_files(filename_list, out_path, outname, inter_time, audio_duration, split_mode):
    """Build the audio_files payload returned to the HTTP layer."""

    audio_files = []
    if split_mode == 1:
        # 返回所有分片音频
        for idx, fname in enumerate(filename_list):
            part_path = os.path.join(WAVS_DIR, fname)
            part_duration = -1
            try:
                part_info = sf.info(part_path)
                part_duration = round(part_info.duration, 2)
            except Exception as e:
                print(f"计算分片音频时长失败: {part_path}, {e}")

            relative_url = f"/static/wavs/{fname}"
            audio_files.append(
                {
                    "filename": part_path,
                    "url": _build_full_url(relative_url),
                    "relative_url": relative_url,
                    "inference_time": round(inter_time, 2),
                    "audio_duration": part_duration,
                    "part_index": idx + 1,
                    "is_merged": 0,
                }
            )

        # 追加合并后的完整音频
        relative_url = f"/static/wavs/{outname}"
        audio_files.append(
            {
                "filename": out_path,
                "url": _build_full_url(relative_url),
                "relative_url": relative_url,
                "inference_time": round(inter_time, 2),
                "audio_duration": audio_duration,
                "is_merged": 1,
            }
        )
    else:
        # 默认保持原行为：仅返回合并后的完整音频
        relative_url = f"/static/wavs/{outname}"
        audio_files.append(
            {
                "filename": out_path,
                "url": _build_full_url(relative_url),
                "relative_url": relative_url,
                "inference_time": round(inter_time, 2),
                "audio_duration": audio_duration,
            }
        )

    return audio_files


def run_tts_generation(
    chat,
    text: str,
    prompt: str,
    voice: str,
    custom_voice: int,
    temperature: float,
    top_p: float,
    top_k: int,
    skip_refine: int,
    is_stream: int,
    speed: int,
    text_seed: int,
    refine_max_new_token: int,
    infer_max_new_token: int,
    split_mode: int,
    segment_len: int,
    merge_size: int,
    device,
):
    """Core TTS generation pipeline.

    This function performs speaker selection, text splitting, inference,
    segment saving, ffmpeg merge, duration calculation, and builds
    the audio_files payload used by the HTTP layer.
    """

    rand_spk, voice_for_filename = _select_speaker(chat, voice, custom_voice, device)

    start_time = time.time()

    grouped_text = _split_and_group_text(text, segment_len, text_seed, merge_size)
    params_infer_code, params_refine_text = _build_infer_params(
        rand_spk,
        speed,
        top_p,
        top_k,
        temperature,
        infer_max_new_token,
        prompt,
        refine_max_new_token,
    )
    print(f"{prompt=}")

    # 在推理时仍然尊重传入的 skip_refine / is_stream 设置
    filename_list, inter_time, audio_time = _infer_and_save_segments(
        chat,
        grouped_text,
        rand_spk,
        params_infer_code,
        params_refine_text,
        text,
        temperature,
        top_p,
        top_k,
        voice_for_filename,
        start_time,
        skip_refine,
        is_stream,
    )

    outname, out_path, audio_duration = _merge_segments(
        filename_list,
        inter_time,
        text,
        voice_for_filename,
        temperature,
        top_p,
        top_k,
        audio_time,
    )

    audio_files = _build_audio_files(
        filename_list,
        out_path,
        outname,
        inter_time,
        audio_duration,
        split_mode,
    )

    return audio_files
