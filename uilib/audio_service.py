import os

import soundfile as sf

from uilib.cfg import WAVS_DIR


def list_audio_files():
    audio_files = []
    if not os.path.exists(WAVS_DIR):
        return audio_files

    for name in os.listdir(WAVS_DIR):
        if not name.lower().endswith(".wav"):
            continue
        full_path = os.path.join(WAVS_DIR, name)
        if not os.path.isfile(full_path):
            continue

        duration = -1
        try:
            info = sf.info(full_path)
            duration = round(info.duration, 2)
        except Exception as e:
            print(f"计算音频时长失败: {full_path}, {e}")

        stat = os.stat(full_path)
        relative_url = f"/static/wavs/{name}"
        audio_files.append(
            {
                "filename": full_path,
                "url": relative_url,
                "relative_url": relative_url,
                "inference_time": 0,
                "audio_duration": duration,
                "mtime": int(stat.st_mtime),
            }
        )

    audio_files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return audio_files


def delete_audio_file(filename: str):
    if not filename:
        raise ValueError("filename is required")

    if "/" in filename or "\\" in filename:
        raise ValueError("invalid filename")

    file_path = os.path.join(WAVS_DIR, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError("file not found")

    os.remove(file_path)
