import json
import logging
import re
from typing import Dict, Tuple, List, Literal, Callable, Optional
import sys

from numba import jit
import numpy as np

from .utils import del_all


@jit
def _find_index(table: np.ndarray, val: np.uint16):
    for i in range(table.size):
        if table[i] == val:
            return i
    return -1


@jit
def _fast_replace(
    table: np.ndarray, text: bytes
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    result = np.frombuffer(text, dtype=np.uint16).copy()
    replaced_words = []
    for i in range(result.size):
        ch = result[i]
        p = _find_index(table[0], ch)
        if p >= 0:
            repl_char = table[1][p]
            result[i] = repl_char
            replaced_words.append((chr(ch), chr(repl_char)))
    return result, replaced_words


class Normalizer:
    def __init__(self, map_file_path: str, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.normalizers: Dict[str, Callable[[str], str]] = {}
        self.homophones_map = self._load_homophones_map(map_file_path)
        """
        homophones_map

        Replace the mispronounced characters with correctly pronounced ones.

        Creation process of homophones_map.json:

        1. Establish a word corpus using the [Tencent AI Lab Embedding Corpora v0.2.0 large] with 12 million entries. After cleaning, approximately 1.8 million entries remain. Use ChatTTS to infer the text.
        2. Record discrepancies between the inferred and input text, identifying about 180,000 misread words.
        3. Create a pinyin to common characters mapping using correctly read characters by ChatTTS.
        4. For each discrepancy, extract the correct pinyin using [python-pinyin] and find homophones with the correct pronunciation from the mapping.

        Thanks to:
        [Tencent AI Lab Embedding Corpora for Chinese and English Words and Phrases](https://ai.tencent.com/ailab/nlp/en/embedding.html)
        [python-pinyin](https://github.com/mozillazg/python-pinyin)

        """
        self.coding = "utf-16-le" if sys.byteorder == "little" else "utf-16-be"
        self.reject_pattern = re.compile(r"[^\u4e00-\u9fffA-Za-z，。、,\. ]")
        self.sub_pattern = re.compile(r"\[uv_break\]|\[laugh\]|\[lbreak\]")
        self.chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
        self.english_word_pattern = re.compile(r"\b[A-Za-z]+\b")
        self.character_simplifier = str.maketrans(
            {
                "：": "，",
                "；": "，",
                "！": "。",
                "（": "，",
                "）": "，",
                "【": "，",
                "】": "，",
                "『": "，",
                "』": "，",
                "「": "，",
                "」": "，",
                "《": "，",
                "》": "，",
                "－": "，",
                ":": ",",
                ";": ",",
                "!": ".",
                "(": ",",
                ")": ",",
                #"[": ",",
                #"]": ",",
                ">": ",",
                "<": ",",
                "-": ",",
            }
        )
        self.halfwidth_2_fullwidth = str.maketrans(
            {
                "!": "！",
                '"': "“",
                "'": "‘",
                "#": "＃",
                "$": "＄",
                "%": "％",
                "&": "＆",
                "(": "（",
                ")": "）",
                ",": "，",
                "-": "－",
                "*": "＊",
                "+": "＋",
                ".": "。",
                "/": "／",
                ":": "：",
                ";": "；",
                "<": "＜",
                "=": "＝",
                ">": "＞",
                "?": "？",
                "@": "＠",
                # '[': '［',
                "\\": "＼",
                # ']': '］',
                "^": "＾",
                # '_': '＿',
                "`": "｀",
                "{": "｛",
                "|": "｜",
                "}": "｝",
                "~": "～",
            }
        )

        # 控制符保护: 识别 ChatTTS 的控制 token, 包括带数字下标的形式
        # 这些 token 需要在归一化/同音字替换过程中完全保留原样
        self.ctrl_pattern = re.compile(
            r"\[(uv_break|laugh|lbreak|break|oral_\d+|laugh_\d+|break_\d+)\]",
            re.I,
        )

    def __call__(
        self,
        text: str,
        do_text_normalization=True,
        do_homophone_replacement=True,
        lang: Optional[Literal["zh", "en"]] = None,
    ) -> str:
        # 如果既不做文本归一化也不做同音字替换，则直接返回原文，避免误删控制符等特殊标记
        if not do_text_normalization and not do_homophone_replacement:
            return text

        # 在做任何内部归一化之前, 先对控制符做占位保护, 避免它们在
        # _count_invalid_characters / reject_pattern / 同音字替换 中被误删或改写。
        protected_text, ctrl_tokens = self._protect_control_tokens(text)

        if do_text_normalization:
            _lang = self._detect_language(protected_text) if lang is None else lang
            if _lang in self.normalizers:
                protected_text = self.normalizers[_lang](protected_text)
            if _lang == "zh":
                protected_text = self._apply_half2full_map(protected_text)

        invalid_characters = self._count_invalid_characters(protected_text)
        if len(invalid_characters):
            self.logger.warning(f"found invalid characters: {invalid_characters}")
            protected_text = self._apply_character_map(protected_text)

        if do_homophone_replacement:
            arr, replaced_words = _fast_replace(
                self.homophones_map,
                protected_text.encode(self.coding),
            )
            if replaced_words:
                protected_text = arr.tobytes().decode(self.coding)
                repl_res = ", ".join([f"{_[0]}->{_[1]}" for _ in replaced_words])
                self.logger.info(f"replace homophones: {repl_res}")

        if len(invalid_characters):
            protected_text = self.reject_pattern.sub("", protected_text)

        # 恢复控制符占位符, 保证输出中控制符仍然是原始形式
        text = self._restore_control_tokens(protected_text, ctrl_tokens)
        return text

    def register(self, name: str, normalizer: Callable[[str], str]) -> bool:
        if name in self.normalizers:
            self.logger.warning(f"name {name} has been registered")
            return False
        try:
            val = normalizer("test string 测试字符串")
            if not isinstance(val, str):
                self.logger.warning("normalizer must have caller type (str) -> str")
                return False
        except Exception as e:
            self.logger.warning(e)
            return False
        self.normalizers[name] = normalizer
        return True

    def unregister(self, name: str):
        if name in self.normalizers:
            del self.normalizers[name]

    def destroy(self):
        del_all(self.normalizers)
        del self.homophones_map

    def _load_homophones_map(self, map_file_path: str) -> np.ndarray:
        with open(map_file_path, "r", encoding="utf-8") as f:
            homophones_map: Dict[str, str] = json.load(f)
        map = np.empty((2, len(homophones_map)), dtype=np.uint32)
        for i, k in enumerate(homophones_map.keys()):
            map[:, i] = (ord(k), ord(homophones_map[k]))
        del homophones_map
        return map

    def _count_invalid_characters(self, s: str):
        s = self.sub_pattern.sub("", s)
        non_alphabetic_chinese_chars = self.reject_pattern.findall(s)
        return set(non_alphabetic_chinese_chars)

    def _apply_half2full_map(self, text: str) -> str:
        return text.translate(self.halfwidth_2_fullwidth)

    def _apply_character_map(self, text: str) -> str:
        return text.translate(self.character_simplifier)

    def _detect_language(self, sentence: str) -> Literal["zh", "en"]:
        chinese_chars = self.chinese_char_pattern.findall(sentence)
        english_words = self.english_word_pattern.findall(sentence)

        if len(chinese_chars) > len(english_words):
            return "zh"
        else:
            return "en"

    # ---- 控制符占位保护相关的内部方法 ----

    def _encode_index(self, i: int) -> str:
        """Encode numeric index into A-Z string, using only letters (安全字符)."""

        if i < 0:
            i = 0
        chars: List[str] = []
        while True:
            chars.append(chr(ord("A") + (i % 26)))
            i //= 26
            if i == 0:
                break
        chars.reverse()
        return "".join(chars)

    def _decode_index(self, tag: str) -> int:
        """Inverse of _encode_index, restore numeric index from A-Z string."""

        idx = 0
        for ch in tag:
            idx = idx * 26 + (ord(ch) - ord("A"))
        return idx

    def _protect_control_tokens(self, text: str) -> Tuple[str, List[str]]:
        """Replace [control] tokens with letter-only placeholders and return (new_text, tokens).

        占位符仅使用 A-Z 字母, 避免被 reject_pattern 当作非法字符删除, 也不会被
        homophones_map 命中, 因为其中只包含中文字符。
        """

        tokens: List[str] = []

        def repl(m: re.Match) -> str:
            full = m.group(0)  # 包含 [ ] 的完整 token
            idx = len(tokens)
            tokens.append(full)
            tag = self._encode_index(idx)
            # 使用字母组成的占位符, 形如 CTRL{TAG}Z, 不包含数字/下划线/中括号
            return f"CTRL{tag}Z"

        protected = self.ctrl_pattern.sub(repl, text)
        return protected, tokens

    def _restore_control_tokens(self, text: str, tokens: List[str]) -> str:
        """Restore CTRL{TAG}Z placeholders back to original [control] tokens."""

        pattern = re.compile(r"CTRL([A-Z]+)Z")

        def repl(m: re.Match) -> str:
            tag = m.group(1)
            idx = self._decode_index(tag)
            if 0 <= idx < len(tokens):
                return tokens[idx]
            return m.group(0)

        return pattern.sub(repl, text)
