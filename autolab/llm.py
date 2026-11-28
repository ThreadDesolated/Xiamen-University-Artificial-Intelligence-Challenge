"""OpenAI 兼容网关的最小客户端：只负责"发提示词、拿回一个 JSON 对象"。

规划器和归因器共用这一层；所有失败都抛 `ValidationError`，由调用方决定回退到
模板/规则，绝不让模型输出直接进入执行链路。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config
from .schemas import ValidationError


def available() -> bool:
    return bool(config.API_KEY)


def extract_json(text: str) -> dict:
    """从模型回复里截取第一个完整 JSON 对象（兼容 ```json 代码块和前后废话）。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    raise ValidationError("模型输出里没有可解析的 JSON 对象")


def chat_json(
    system: str,
    user: str,
    *,
    timeout: int = 30,
    max_tokens: int = 900,
    temperature: float = 0.2,
) -> dict:
    """调用 chat/completions 并返回解析后的 JSON；未配置 Key 直接报错。"""
    if not config.API_KEY:
        raise ValidationError("未配置 DEEPSEEK_API_KEY，无法调用大模型")

    payload = {
        "model": config.MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{config.BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except OSError:
            pass
        raise ValidationError(f"大模型返回 HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValidationError(f"大模型调用失败: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError("大模型返回结构异常（缺少 choices/message/content）") from exc
    return extract_json(str(content))
