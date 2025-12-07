import os

import ChatTTS
from ChatTTS.norm import Normalizer

from uilib import utils


SAMPLE_TEXT = """hello当前遇到的问题 [uv_break] 就是:
python app.py 启动 [oral_2] 之后,
我通过 frp 将本地服务暴露到公网 [laugh_0] 上
然后通过域名方向代理 [break_6] 访问公网ip和服务端口,成功访问.
并且也成功生成了 [laugh_4] 音频文件.
但是无法 [oral_2] 下载音频文件
"""


def test_remove_brackets_preserves_control_tokens_with_digits():
    line = "测试 [oral_2] [laugh_0] [break_6] [uv_break]"
    out = utils.remove_brackets(line)

    # remove_brackets should keep all control tokens in [token] form
    for tok in ["[oral_2]", "[laugh_0]", "[break_6]", "[uv_break]"]:
        assert tok in out, f"token {tok} should be preserved, got: {out!r}"


def test_split_text_preserves_control_tokens_in_segments():
    # Simulate the way app.py/_split_and_group_text splits text into lines
    lines = [ln for ln in SAMPLE_TEXT.splitlines() if ln.strip()]

    segments = utils.split_text(lines, segment_len=0)
    joined = " ".join(segments)

    # All control tokens from SAMPLE_TEXT must still exist after split/normalize
    for tok in ["[uv_break]", "[oral_2]", "[laugh_0]", "[laugh_4]", "[break_6]"]:
        assert tok in joined, f"token {tok} should be present in split text, got: {joined!r}"


def test_chattts_normalizer_keeps_control_tokens_when_no_normalization():
    # Ensure our patch to ChatTTS.Normalizer does not treat control tokens as invalid
    homophones_path = os.path.join(os.path.dirname(ChatTTS.__file__), "res", "homophones_map.json")
    norm = Normalizer(homophones_path)

    text = "[oral_2] [laugh_0] [break_6] [uv_break]"
    out = norm(text, do_text_normalization=False, do_homophone_replacement=False)

    for tok in ["[oral_2]", "[laugh_0]", "[break_6]", "[uv_break]"]:
        assert tok in out, f"token {tok} should be preserved by Normalizer, got: {out!r}"


def test_chattts_normalizer_keeps_control_tokens_with_homophones_enabled():
    homophones_path = os.path.join(os.path.dirname(ChatTTS.__file__), "res", "homophones_map.json")
    norm = Normalizer(homophones_path)

    text = "[oral_2] [laugh_0] [break_6] [uv_break]"
    out = norm(text, do_text_normalization=False, do_homophone_replacement=True)

    for tok in ["[oral_2]", "[laugh_0]", "[break_6]", "[uv_break]"]:
        assert tok in out, f"token {tok} should be preserved with homophones enabled, got: {out!r}"
