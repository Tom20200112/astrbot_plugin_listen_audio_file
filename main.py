"""听音频文件：让 bot 能听聊天里别人发的音频文件（mp3/m4a/wav/amr…）和语音条。

AstrBot 自己不看「文件」消息（群聊上下文里直接丢掉）；语音条最多被语音转文字转成文字，听不出唱得怎么样。
这里做三件事：
1. 盯着所有消息，谁发了音频文件或语音条就记下来（顺手存一份到本地，防链接过期/文件被删）；
2. bot 被叫到时，在提示词里提一句「最近有人发了音频，想听就调工具」；
3. 提供 listen_audio_file 工具，真正去听；另有 /听音频 命令方便手动测试。
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from collections import deque
from dataclasses import dataclass

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .audio_core import (
    DEFAULT_PROMPT,
    AudioError,
    ask_model,
    fetch_bytes,
    is_audio_name,
    is_local_audio_file,
    sniff_audio_ext,
    to_small_mp3,
)

PLUGIN_NAME = "astrbot_plugin_listen_audio_file"
VOICE_LABEL = "语音条"


@dataclass
class AudioRecord:
    name: str
    url: str
    sender: str
    ts: float
    group_id: str = ""
    file_id: str = ""
    local_path: str = ""
    kind: str = "file"  # file＝音频文件；voice＝语音条
    msg_id: str = ""
    record_file: str = ""  # 语音条在 QQ 端的文件名（找 QQ 端重取用）

    @property
    def label(self) -> str:
        return VOICE_LABEL if self.kind == "voice" else f"音频文件「{self.name}」"


def _http(url) -> str:
    url = str(url or "")
    return url if url.startswith(("http://", "https://")) else ""


def _want_kind(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if "语音" in k or "voice" in k:
        return "voice"
    if "文件" in k or "file" in k:
        return "file"
    return ""


class ListenAudioFile(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self.recent: dict[str, deque[AudioRecord]] = {}
        try:
            base = str(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception:
            base = os.path.join("data", "plugin_data", PLUGIN_NAME)
        self.cache_dir = os.path.join(base, "audio_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---------- 配置 ----------
    def _cfg(self, key: str, default):
        value = self.config.get(key, default)
        return default if value in (None, "") else value

    @property
    def _max_bytes(self) -> int:
        return int(self._cfg("max_file_mb", 40)) * 1024 * 1024

    @property
    def _allow_private(self) -> bool:
        return bool(self._cfg("allow_private_network", False))

    # ---------- 1. 记下每个音频文件和语音条 ----------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def remember_audio_files(self, event: AstrMessageEvent):
        include_voice = bool(self._cfg("include_voice", True))
        raw_ids = self._raw_file_ids(event)
        sender = event.get_sender_name() or event.get_sender_id()
        for comp in event.get_messages():
            if isinstance(comp, Comp.Record) and include_voice:
                self._remember_voice(event, comp, sender)
            elif isinstance(comp, Comp.File) and is_audio_name(comp.name):
                rec = AudioRecord(
                    name=comp.name, url=comp.url or "", sender=sender, ts=time.time(),
                    group_id=event.get_group_id() or "", file_id=raw_ids.get(comp.name, ""),
                    local_path=comp.file_ if comp.file_ and os.path.isfile(comp.file_) else "",
                    msg_id=self._msg_id(event),
                )
                self._remember(event, rec)
                logger.info(f"[听音频文件] 记下 {rec.sender} 发的「{rec.name}」")
                if self._cfg("cache_on_arrival", True) and not rec.local_path:
                    asyncio.create_task(self._save_copy(rec))

    def _remember(self, event: AstrMessageEvent, rec: AudioRecord) -> None:
        self.recent.setdefault(event.unified_msg_origin, deque(maxlen=50)).append(rec)

    @staticmethod
    def _msg_id(event: AstrMessageEvent) -> str:
        try:
            return str(event.message_obj.message_id or "")
        except Exception:
            return ""

    def _remember_voice(self, event: AstrMessageEvent, comp: Comp.Record, sender: str) -> None:
        """语音条：AstrBot 预处理时已经把它下到本机并转成 wav（临时文件，事件结束就删），趁现在复制一份。"""
        rec = AudioRecord(
            name=VOICE_LABEL, url=_http(comp.url), sender=sender, ts=time.time(),
            group_id=event.get_group_id() or "", kind="voice", msg_id=self._msg_id(event),
            record_file=self._raw_record_file(event),
        )
        self._remember(event, rec)
        logger.debug(f"[听音频文件] 记下 {sender} 的语音条")
        if self._cfg("cache_on_arrival", True):
            self._copy_voice(rec, comp)

    def _copy_voice(self, rec: AudioRecord, comp: Comp.Record) -> None:
        src = next((str(p) for p in (comp.path, comp.file) if p and is_local_audio_file(str(p))), "")
        if not src:
            return
        try:
            with open(src, "rb") as f:
                data = f.read(self._max_bytes + 1)
            if not data or len(data) > self._max_bytes:
                return
            mid = re.sub(r"[^0-9-]", "", rec.msg_id) or str(int(rec.ts * 1000))
            path = os.path.join(self.cache_dir, f"{mid}_voice{sniff_audio_ext(data) or '.bin'}")
            with open(path, "wb") as f:
                f.write(data)
            rec.local_path = path
        except Exception as exc:
            logger.info(f"[听音频文件] 存语音条失败（不影响之后再取）：{exc}")
        self._prune_cache()

    @staticmethod
    def _raw_segments(event: AstrMessageEvent) -> list:
        try:
            return list(event.message_obj.raw_message.get("message") or [])
        except Exception:
            return []

    def _raw_file_ids(self, event: AstrMessageEvent) -> dict[str, str]:
        """QQ 原始消息里的 file_id，链接过期后可以用它重新要下载地址（NapCat 写 file_id，SnowLuma 还写了 id）。"""
        out: dict[str, str] = {}
        for seg in self._raw_segments(event):
            if isinstance(seg, dict) and seg.get("type") == "file":
                data = seg.get("data") or {}
                out[str(data.get("file") or data.get("name") or "")] = str(data.get("file_id") or data.get("id") or "")
        return out

    def _raw_record_file(self, event: AstrMessageEvent) -> str:
        for seg in self._raw_segments(event):
            if isinstance(seg, dict) and seg.get("type") == "record":
                return str((seg.get("data") or {}).get("file") or "")
        return ""

    async def _save_copy(self, rec: AudioRecord) -> None:
        try:
            data = await fetch_bytes(rec.url, self._max_bytes, allow_private=self._allow_private)
            safe = re.sub(r"[^\w.\-]+", "_", rec.name or "")[-80:]
            path = os.path.join(self.cache_dir, f"{int(rec.ts)}_{safe}")
            with open(path, "wb") as f:
                f.write(data)
            rec.local_path = path
        except Exception as exc:
            logger.info(f"[听音频文件] 预存「{rec.name}」失败（不影响之后现下）：{exc}")
        self._prune_cache()

    def _prune_cache(self) -> None:
        cutoff = time.time() - int(self._cfg("keep_days", 7)) * 86400
        for fn in self._safe_listdir():
            path = os.path.join(self.cache_dir, fn)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass

    # ---------- 2. 被叫到时提一句 ----------
    @filter.on_llm_request()
    async def hint_recent_audio(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._cfg("hint_in_prompt", True):
            return
        cutoff = time.time() - 6 * 3600
        recs = [r for r in self.recent.get(event.unified_msg_origin, []) if r.ts >= cutoff][-3:]
        if not recs:
            return
        lines = "\n".join(
            f"- {r.sender} 在 {time.strftime('%H:%M', time.localtime(r.ts))} 发的"
            + (VOICE_LABEL if r.kind == "voice" else f"「{r.name}」")
            for r in recs
        )
        req.system_prompt = (req.system_prompt or "") + (
            "\n\n[音频] 这个聊天最近有人发了下面这些音频：\n"
            f"{lines}\n"
            "音频文件不会自动听；语音条最多自动转成文字，听不出唱得怎么样。"
            "想知道里面是什么，或者有人让你听、让你点评时，用 listen_audio_file 工具去听。听语音条时，kind 填「语音条」。"
        )

    # ---------- 3. 真正去听 ----------
    @filter.llm_tool(name="listen_audio_file")
    async def listen_audio_file(self, event: AstrMessageEvent, file_name: str = "", focus: str = "", kind: str = "") -> str:
        """听聊天里别人发的音频文件（mp3/m4a/wav/amr 等）或语音条。有人发了音频、想让你听听里面是什么、或让你点评唱得怎么样时调用。语音条最多被自动转成文字，要听唱得怎么样、男声女声、语气就调这个工具。回复（引用）某条音频再叫你时，听的就是那一条。

        Args:
            file_name(string): 要听的音频文件名，写一部分也行；不填就听最近的
            focus(string): 这次重点听什么，例如「唱的哪首歌、音准气息怎么样」「逐句听写歌词」；不填按通用角度听
            kind(string): 想听语音条就填「语音条」，想听音频文件就填「文件」；不填就听最近的一个（两种都算）
        """
        try:
            rec, text = await self._listen(event, file_name, focus, kind)
        except AudioError as exc:
            return f"没听成：{exc}。如实告诉对方你这次没听到，别编听感。"
        except Exception as exc:
            logger.warning(f"[听音频文件] 出错：{type(exc).__name__}: {exc}")
            return "没听成（插件内部出错）。如实告诉对方你这次没听到，别编听感。"
        return (
            f"（你刚听了 {rec.sender} 发的{rec.label}）\n{text}\n"
            "——以上是你自己听出来的，用你自己的话讲给对方；"
            "标了「不确定/听不出」的地方别说死，歌名原唱没把握就别断言。"
        )

    @filter.command("听音频")
    async def listen_command(self, event: AstrMessageEvent):
        """手动测试：/听音频 [想重点听什么]。回复一个音频文件或语音条发这条就听那个，否则听最近一个。"""
        focus = re.sub(r"^\s*/?听音频", "", event.message_str or "").strip()
        try:
            rec, text = await self._listen(event, "", focus, "")
            title = f"{rec.sender} 的语音条" if rec.kind == "voice" else rec.name
            yield event.plain_result(f"🎧 {title}\n{text}")
        except AudioError as exc:
            yield event.plain_result(f"没听成：{exc}")
        except Exception as exc:
            logger.warning(f"[听音频文件] 出错：{type(exc).__name__}: {exc}")
            yield event.plain_result(f"没听成（插件内部出错：{type(exc).__name__}）")

    # ---------- 找文件、取字节、送模型 ----------
    def _pick(self, event: AstrMessageEvent, file_name: str, kind: str) -> tuple[AudioRecord | None, bool]:
        """返回 (要听的那条, file_name 有没有对上)。先看这条消息本身/引用的那条，再看最近记下的。"""
        sender = event.get_sender_name() or event.get_sender_id()
        want = _want_kind(kind)
        for comp in event.get_messages():
            if isinstance(comp, Comp.File) and is_audio_name(comp.name) and want != "voice":
                return self._from_file_comp(comp, sender, event), True
            if isinstance(comp, Comp.Record) and want != "file":
                return self._from_record_comp(comp, sender, event, msg_id=self._msg_id(event)), True
            if isinstance(comp, Comp.Reply):
                who = comp.sender_nickname or "对方"
                for sub in comp.chain or []:
                    if isinstance(sub, Comp.File) and is_audio_name(sub.name) and want != "voice":
                        return self._from_file_comp(sub, who, event), True
                    if isinstance(sub, Comp.Record) and want != "file":
                        return self._from_record_comp(sub, who, event, msg_id=str(comp.id or "")), True
        recs = [r for r in self.recent.get(event.unified_msg_origin, []) if not want or r.kind == want]
        if file_name:
            key = file_name.strip().lower()
            hits = [r for r in recs if r.kind == "file" and key in r.name.lower()]
            if hits:
                return hits[-1], True
            return (recs[-1] if recs else None), False
        return (recs[-1] if recs else None), True

    def _from_file_comp(self, comp: Comp.File, sender: str, event: AstrMessageEvent) -> AudioRecord:
        # 之前记过的就用那条（可能已经存了本地副本）
        for r in reversed(self.recent.get(event.unified_msg_origin, [])):
            if r.kind == "file" and r.name == comp.name:
                return r
        local = comp.file_ if comp.file_ and os.path.isfile(comp.file_) else self._find_saved(comp.name)
        return AudioRecord(
            name=comp.name, url=comp.url or "", sender=sender, ts=time.time(),
            group_id=event.get_group_id() or "", local_path=local,
        )

    def _from_record_comp(self, comp: Comp.Record, sender: str, event: AstrMessageEvent, msg_id: str = "") -> AudioRecord:
        if msg_id:
            for r in reversed(self.recent.get(event.unified_msg_origin, [])):
                if r.kind == "voice" and r.msg_id == msg_id:
                    return r
        rec = AudioRecord(name=VOICE_LABEL, url=_http(comp.url), sender=sender, ts=time.time(),
                          group_id=event.get_group_id() or "", kind="voice", msg_id=msg_id)
        src = next((str(p) for p in (comp.path, comp.file) if p and is_local_audio_file(str(p))), "")
        if src:
            rec.local_path = src  # AstrBot 预处理下好的临时文件，这次事件里还在
        elif msg_id:
            prefix = f"{re.sub(r'[^0-9-]', '', msg_id)}_voice"
            saved = [fn for fn in self._safe_listdir() if fn.startswith(prefix)]
            if saved:
                rec.local_path = os.path.join(self.cache_dir, saved[0])
        return rec

    def _safe_listdir(self) -> list[str]:
        try:
            return os.listdir(self.cache_dir)
        except OSError:
            return []

    def _find_saved(self, name: str) -> str:
        """重启后名单清空了，但存下的副本还在：按文件名找最新的那份。"""
        suffix = "_" + re.sub(r"[^\w.\-]+", "_", name or "")[-80:]
        hits = [os.path.join(self.cache_dir, fn) for fn in self._safe_listdir() if fn.endswith(suffix)]
        return max(hits, key=os.path.getmtime) if hits else ""

    async def _call_qq(self, event: AstrMessageEvent, action: str, **params):
        return await event.bot.call_action(action=action, self_id=int(event.get_self_id()), **params)

    async def _get_voice_bytes(self, event: AstrMessageEvent, rec: AudioRecord) -> bytes:
        errors = []
        for src in (rec.local_path, rec.url):
            if not src:
                continue
            try:
                return await fetch_bytes(src, self._max_bytes, allow_private=self._allow_private)
            except AudioError as exc:
                errors.append(str(exc))
        # 没副本、链接也不行：让 QQ 端把语音再给一次（只认最近几小时的）
        if rec.record_file and event.get_platform_name() == "aiocqhttp":
            try:
                got = await self._call_qq(event, "get_record", file=rec.record_file, out_format="mp3")
                if isinstance(got, dict):
                    if got.get("base64"):
                        return base64.b64decode(str(got["base64"]))
                    path = str(got.get("file") or "")
                    if is_local_audio_file(path):  # 只读确实是音频的本地文件
                        return await fetch_bytes(path, self._max_bytes)
                    if _http(got.get("url")):
                        return await fetch_bytes(got["url"], self._max_bytes, allow_private=self._allow_private)
            except AudioError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"找 QQ 端重取语音失败：{exc}")
        raise AudioError(errors[-1] if errors else "这条语音条取不回来了，可能是发得太久了")

    async def _get_bytes(self, event: AstrMessageEvent, rec: AudioRecord) -> bytes:
        if rec.kind == "voice":
            return await self._get_voice_bytes(event, rec)
        errors = []
        for src in (rec.local_path, rec.url):
            if not src:
                continue
            try:
                return await fetch_bytes(src, self._max_bytes, allow_private=self._allow_private)
            except AudioError as exc:
                errors.append(str(exc))
        # 链接过期了：QQ 群文件还在的话，重新要一个下载地址
        if rec.file_id and rec.group_id and event.get_platform_name() == "aiocqhttp":
            try:
                ret = await self._call_qq(event, "get_group_file_url", group_id=int(rec.group_id), file_id=rec.file_id)
                url = _http(ret.get("url")) if isinstance(ret, dict) else ""
                if url:  # 只认网址，免得 QQ 端返回怪东西时去读本机文件
                    return await fetch_bytes(url, self._max_bytes, allow_private=self._allow_private)
                errors.append("群文件重取没拿到下载地址")
            except AudioError as exc:
                errors.append(str(exc))
            except Exception as exc:
                errors.append(f"群文件重取失败：{exc}")
        raise AudioError(errors[-1] if errors else "拿不到这个文件，可能已被删除或过期")

    async def _listen(self, event: AstrMessageEvent, file_name: str, focus: str, kind: str) -> tuple[AudioRecord, str]:
        rec, matched = self._pick(event, file_name, kind)
        if not matched and file_name and not focus:
            # 模型偶尔把「想重点听什么」错填进文件名：对不上任何文件，就当成 focus
            logger.info(f"[听音频文件] 文件名「{file_name[:40]}」对不上，当成想重点听的内容")
            focus = file_name
        if rec is None:
            raise AudioError("这个聊天里最近没有看到音频文件或语音条。插件只记得装好之后发的")
        raw = await self._get_bytes(event, rec)
        fname = rec.name
        if rec.kind == "voice":
            ext = sniff_audio_ext(raw)
            if ext == ".silk":
                raise AudioError("这条语音还是 QQ 原始的 silk 格式，插件转不了")
            fname = f"voice{ext}"
        audio, fmt = await to_small_mp3(raw, fname, int(self._cfg("max_seconds", 600)))
        prompt = DEFAULT_PROMPT
        if focus:
            prompt = (
                f"听这段音频，按下面的要求回答，简洁具体、别说套话：\n{focus}\n"
                "歌名/原唱这类事实不确定就明说不确定，不要猜；点评要把被评的那句引出来。"
            )
        text = await ask_model(
            audio, fmt, prompt,
            base_url=str(self._cfg("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            api_key=str(self._cfg("api_key", "")).strip(),
            model=str(self._cfg("model", "qwen3.5-omni-flash")),
            timeout=float(self._cfg("timeout_seconds", 90)),
        )
        return rec, text
