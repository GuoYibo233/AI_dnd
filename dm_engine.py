#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 DeepSeek API 的 DM 引擎。

核心理念：
1. 所有文本（旁白、NPC 发言）都由模型产出，工具只提供上下文与状态。
2. DM 和 NPC 分别拥有独立的 prompt 与日志，每次调用都会把完整输入/输出写入
   stories/<campaign_id>/<conversation_id>/logs/<agent>/<agent_id>/timestamp.json。
3. 每场对话（conversation_id）都有独立的 npcs 目录：stories/<campaign_id>/<conversation_id>/npcs/，
   退场时 NPC 的记忆会写入该目录，之后同一场对话再次唤出时会读取对应文件。
4. DM 可以在输出中自荐是否需要切换到 DeepSeek-R1 做更深的推理。
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


SOPHNET_ENDPOINT = "https://www.sophnet.com/api/open-apis/v1/chat/completions"
DM_DEFAULT_MODEL = "DeepSeek-V3.1"
DM_REASONING_MODEL = "DeepSeek-R1"

# 临时开场白配置
USE_FIXED_OPENING = True
FIXED_OPENING_TEXT = (
    "黄昏时分，雨滴断断续续地敲打着余烬村的屋顶和石板路。北境山麓的寒风穿过黑松林，"
    "带来潮湿的松脂和泥土的气息。村庄被深谷与密林环抱，只有几盏挂在屋檐下的油灯在雨中摇曳，"
    "投下昏黄的光晕。矿脉早已沉寂，铁匠铺和木工坊也早早收了工，但村民们仍在为明天的灰烬灯节忙碌着。\n\n"
    "村中央的广场上，那座古老的石台静静伫立，雨水顺着石缝流淌。石台顶端，那盏由日耀钢打造的灯芯尚未点燃，"
    "却在雨幕中隐隐反射着灯笼的微光。按照传统，明天夜里，它将由牧师祝福，燃起如白昼般的光芒，"
    "成为村民整个寒冬的希望之火——护佑炉灶不熄，驱散暗夜中潜行的野兽。\n\n"
    "可此刻，乌云低垂，月光全无，只有雨声和偶尔从河谷传来的风声，衬得这座山村格外孤寂。\n\n"
    "黄昏时分，雨丝时缓时急地敲打着村庄的屋檐。厚重的云层吞没了月光，只有零星悬挂的灯笼在湿漉漉的街道上投下摇曳的光晕。"
    "就在这样的天气里，奥术师赛伦·弧光踏入了村庄。他是应老村长的请求而来——近两个月，村长寄往王都星辉学院的信件中，"
    "屡次提到一些细微却令人不安的异常，担心它们会干扰即将到来的古老仪式。学院因此派出赛伦，任务明确："
    "观察仪式进程，记录当地的习俗，并在魔法相关的事务上提供必要的协助。雨声淅沥中，他的到来仿佛一束微光，"
    "悄然没入这个被忧虑与期待笼罩的节日前夜。\n\n"
    "雨丝斜织，暮色沉沉。赛伦站在余烬村饱经风霜的木牌坊下，雨水顺着斗篷的褶皱滑落，浸湿了脚下泥泞的土地。"
    "他怀揣着学院那封盖有暗红封蜡的信件，行囊简朴，风尘仆仆。牌坊的阴影里，一点昏黄的光晕由远及近，摇曳着破开雨幕——"
    "是老村长奥尔德·灰杉。他举着一盏旧油灯，微佝着背，灯玻璃上凝结的水汽让那光芒显得格外温暖。\n\n"
    "“路上辛苦了，”村长将一枚冰凉的铜质徽记塞进赛伦手中，上面刻着的灰烬灯图案在灯下泛着微光。“我们先避避雨。”\n\n"
    "老人转身引路，油灯的光圈在湿漉漉的石板路上跳动，照亮了坑洼处积聚的雨水。他们沿着村庄的主路缓缓前行，"
    "鞋底踏在石板上发出沉闷的声响。奥尔德村长不时停下，用提着灯的手指向沿途的重要地点：被雨水冲刷得发亮的广场石板、"
    "铁匠铺里隐约传来炉火熄灭后的余温气息、礼拜堂尖顶沉默的轮廓，以及更远处，那片在暮霭与雨帘中显得愈发深邃、"
    "仿佛吞噬光线的黑松林。他语速平缓，夹杂着雨声，简要交代着明日灰烬灯节的流程，言语间却透出一丝不易察觉的沉重，"
    "暗示着村庄正面临的某种压力。\n\n"
    "最终，他们停在村庄广场边缘一处简陋却结实的遮雨棚下。雨水从棚檐成串滴落，在脚边溅起细小的水花。棚内干燥，隔绝了外面的风雨声。"
    "奥尔德村长收起油灯，指了指棚外被几盏灯笼朦胧照亮的广场和周围建筑的影子。\n\n"
    "“就在这里歇歇脚吧，”他说道，“你可以先熟悉一下环境。想想接下来，最想先见见谁，或者去哪个地方看看。”"
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class NPCProfile:
    npc_id: str
    name: str
    first_impression: str
    roleplay_notes: str
    daily_routine: str
    source: str  # seed_npc / seed_primary / seed_supporting / generated
    dm_instruction: str = ""
    key_information: List[str] = field(default_factory=list)


@dataclass
class LocationState:
    location_id: str
    name: str
    description: str
    connections: List[str]
    npcs: List[str] = field(default_factory=list)
    players: List[str] = field(default_factory=list)


@dataclass
class WorldState:
    time: str
    weather: str
    locations: Dict[str, LocationState]
    npc_profiles: Dict[str, NPCProfile] = field(default_factory=dict)
    players: List[Dict[str, Any]] = field(default_factory=list)
    active_npcs: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NPCContext:
    profile: NPCProfile
    private_memory: List[str] = field(default_factory=list)
    recent_dialogue: List[str] = field(default_factory=list)

    def remember(self, note: str) -> None:
        if note:
            self.private_memory.append(note)

    def log_dialogue(self, text: str) -> None:
        if text:
            self.recent_dialogue.append(text)
            if len(self.recent_dialogue) > 12:
                self.recent_dialogue = self.recent_dialogue[-12:]


# ---------------------------------------------------------------------------
# API 客户端
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, api_key_env: str = "SOPHNET_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError("未找到 SOPHNET_API_KEY，请在环境中配置。")
        self.api_key = api_key

    def call(self, messages: List[Dict[str, str]], model: str) -> str:
        payload = {"messages": messages, "model": model}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(SOPHNET_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"API 返回格式异常: {data}") from exc


# ---------------------------------------------------------------------------
# DM 引擎
# ---------------------------------------------------------------------------

class DMEngine:
    def __init__(
        self,
        story_dir: str,
        *,
        seed_filename: str = "story_seed.json",
        conversation_id: Optional[str] = None,
    ):
        self.story_dir = Path(story_dir)
        if not self.story_dir.exists():
            raise ValueError(f"故事目录不存在：{self.story_dir}")
        if not self.story_dir.is_dir():
            raise ValueError(f"story_dir 必须是目录：{self.story_dir}")
        self.story_seed_path = self.story_dir / seed_filename
        if not self.story_seed_path.exists():
            raise ValueError(f"未找到剧本种子：{self.story_seed_path}")
        self.story_seed = self._load_seed()
        self.players = self._load_players()
        self.player_display_name = self.players[0]["name"] if self.players else "玩家"
        self.npc_catalog = self._build_npc_catalog()
        self.active_contexts: Dict[str, NPCContext] = {}
        self.world_log: List[Dict[str, Any]] = []
        self.current_location = self._resolve_start_location()
        self.world_state = self._build_initial_world_state()
        self.turn_counter = 0

        campaign_id = self.story_seed["world_meta"].get("campaign_id", self.story_dir.name)
        self.campaign_id = campaign_id
        self.conversation_id = conversation_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # session_id 沿用旧字段名，便于外部引用
        self.session_id = self.conversation_id

        self.conversation_root = self.story_dir / self.conversation_id
        self.log_root = self.conversation_root / "logs"
        self.dm_log_root = self.log_root / "dm"
        self.dm_log_dir = self.dm_log_root / "main"
        self.npc_log_root = self.log_root / "npc"
        self.npc_runtime_dir = self.conversation_root / "npcs"
        for path in [self.dm_log_dir, self.npc_log_root, self.npc_runtime_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.llm = LLMClient()
        self.dm_model = DM_DEFAULT_MODEL
        self.narrator_model = DM_DEFAULT_MODEL
        self._transcript_cache: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 数据加载
    def _load_seed(self) -> Dict[str, Any]:
        return json.loads(self.story_seed_path.read_text(encoding="utf-8"))

    def _load_players(self) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for path in sorted(self.story_dir.glob("pc_*.json")):
            cards.append(json.loads(path.read_text(encoding="utf-8")))
        return cards

    def _build_initial_world_state(self) -> Dict[str, Any]:
        """
        从 story_seed 构造初始的世界状态：
        - time / weather：基于 world_meta.opening.time_and_weather 做一个简单拆分；
        - locations：与 story_seed.locations 保持字段同步，并为每个地点预留 npcs 列表。
        """
        world_meta = self.story_seed.get("world_meta", {})
        opening = world_meta.get("opening", {})
        time_and_weather = opening.get("time_and_weather", "")

        time_text = time_and_weather
        weather_text = ""
        if "，" in time_and_weather:
            first, rest = time_and_weather.split("，", 1)
            time_text = first
            weather_text = rest

        locations_seed = self.story_seed.get("locations", {})
        locations_state: Dict[str, Dict[str, Any]] = {}
        for loc_id, payload in locations_seed.items():
            locations_state[loc_id] = {
                "name": payload.get("name", loc_id),
                "description": payload.get("description", ""),
                "connections": list(payload.get("connections", [])),
                "npcs": [],
                "players": [],
            }

        npc_profiles_state: Dict[str, Dict[str, Any]] = {}
        for npc_id, profile in self.npc_catalog.items():
            npc_profiles_state[npc_id] = {
                "name": profile.name,
                "first_impression": profile.first_impression,
                "roleplay_notes": profile.roleplay_notes,
                "daily_routine": profile.daily_routine,
                "dm_instruction": getattr(profile, "dm_instruction", ""),
                "key_information": list(profile.key_information),
            }

        players_state: List[Dict[str, Any]] = []
        for card in self.players:
            player_id = card.get("character_id") or card.get("id") or card.get("name", "")
            players_state.append(
                {
                    "player_id": player_id,
                    "name": card.get("name", ""),
                    "class": card.get("class", ""),
                    "alignment": card.get("alignment", ""),
                    "concept_summary": card.get("concept_summary", ""),
                }
            )

        starting_location = getattr(self, "current_location", None) or opening.get("location")
        if starting_location and starting_location in locations_state:
            locations_state[starting_location]["players"] = [p["player_id"] for p in players_state]

        return {
            "time": time_text,
            "weather": weather_text,
            "locations": locations_state,
            "npc_profiles": npc_profiles_state,
            "players": players_state,
            "active_npcs": [],
            "recent_events": [],
        }

    def _resolve_start_location(self) -> str:
        meta = self.story_seed["world_meta"]
        return meta.get("starting_location") or meta.get("starting_scene_id") or "village_square"

    def _build_npc_catalog(self) -> Dict[str, NPCProfile]:
        catalog: Dict[str, NPCProfile] = {}
        if "npcs" in self.story_seed:
            for npc_id, payload in self.story_seed["npcs"].items():
                catalog[npc_id] = NPCProfile(
                    npc_id=npc_id,
                    name=payload["name"],
                    first_impression=payload.get("first_impression", ""),
                    roleplay_notes=payload.get("roleplay_notes", ""),
                    daily_routine=payload.get("daily_routine", ""),
                    source="seed_npc",
                    dm_instruction=payload.get("dm_instruction", "") or "",
                    key_information=payload.get("key_information", []) or []
                )
        else:
            for section, source in (("primary_npcs", "seed_primary"), ("supporting_npcs", "seed_supporting")):
                for npc_id, payload in self.story_seed.get(section, {}).items():
                    catalog[npc_id] = NPCProfile(
                        npc_id=npc_id,
                        name=payload["name"],
                        first_impression=payload.get("first_impression", ""),
                        roleplay_notes=payload.get("roleplay_notes", ""),
                        daily_routine=payload.get("daily_routine", ""),
                        source=source,
                        key_information=payload.get("key_information", []) or []
                    )
        return catalog

    # ------------------------------------------------------------------
    # 日志工具
    def _write_agent_log(self, agent_type: str, agent_id: str, *, system_prompt: str, user_payload: Any, response: str, model: str) -> None:
        timestamp = datetime.utcnow().strftime("%H%M%S_%f")
        if agent_type == "dm":
            directory = self.dm_log_dir if agent_id == "main" else (self.dm_log_root / agent_id)
        else:
            directory = self.npc_log_root / agent_id
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"{timestamp}.json"
        record = {
            "model": model,
            "system": system_prompt,
            "user": user_payload,
            "response": response
        }
        log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_world_event(self, entry: Dict[str, Any]) -> None:
        self.world_log.append(entry)
        self._transcript_cache.append(entry)

    # ------------------------------------------------------------------
    # Prompt 构建（DM = 管理者，不写文案）
    def _build_simplified_world_state(self) -> Dict[str, Any]:
        """
        构建简化版的 world_state，供 DM AI 使用。
        NPC profiles 中移除 roleplay_notes 和 key_information。
        
        Returns:
            Dict[str, Any]: 简化后的 world_state 深拷贝，保持原始 world_state 不变。
        """
        snapshot = deepcopy(self.world_state)
        simplified_profiles: Dict[str, Dict[str, Any]] = {}
        for npc_id, profile in snapshot.get("npc_profiles", {}).items():
            if not isinstance(profile, dict):
                continue
            simplified_profiles[npc_id] = {
                key: value
                for key, value in profile.items()
                if key not in {"roleplay_notes", "key_information"}
            }
        snapshot["npc_profiles"] = simplified_profiles
        return snapshot

    def _build_dm_messages(self, player_text: str) -> tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        world_meta = self.story_seed["world_meta"]
        location_block = self.story_seed["locations"].get(self.current_location, {})
        recent_events = self.world_log[-6:]
        active_npcs_info = [
            {
                "npc_id": ctx.profile.npc_id,
                "name": ctx.profile.name,
                "memory": ctx.private_memory,
                "recent_dialogue": ctx.recent_dialogue[-3:]
            }
            for ctx in self.active_contexts.values()
        ]
        players_brief = [
            {
                "name": card["name"],
                "class": card["class"],
                "alignment": card.get("alignment", ""),
                "concept_summary": card.get("concept_summary", "")
            }
            for card in self.players
        ]
        catalog_preview = [
            {
                "npc_id": npc_id,
                "name": profile.name,
                "first_impression": profile.first_impression,
                "daily_routine": profile.daily_routine,
                "dm_instruction": profile.dm_instruction,
            }
            for npc_id, profile in list(self.npc_catalog.items())[:6]
        ]

        system_prompt = textwrap.dedent(
            """
            你是一个桌面角色扮演 / 互动叙事系统中的「DM 管理 AI」。

            你的唯一职责：在每一回合，根据玩家输入与当前局面，决策“幕后操作”，并以结构化 JSON 指示其他模块该做什么。你只做调度与控制，不生成任何叙事文本或台词。

            【硬性输出要求】

            1. 你的输出必须且只允许是一个合法 JSON 对象：
            - 不得输出任何额外文字、注释、解释。
            - 不得使用 markdown、代码块标记（如 ```json）。
            - 不得输出 JSON 数组作为最外层结构。

            2. JSON 结构（字段必须按此含义使用）：

            {
                "narration_request": {
                    "need_narration": true/false,
                    // 若 need_narration 为 true，必须提供以下字段且为非空字符串；
                    // 若 need_narration 为 false，则 focus 和 tone 必须为 null。
                    "focus": "本回合旁白应描述/强调的要点",
                    "tone": "旁白语气，如“紧张但克制”"
                },
                "npc_control": {
                    "actions": [
                    {
                        "action": "enter" | "speak" | "exit" | "create",
                        "npc_id": "已有 NPC 的唯一 id；若为 create 可为 null 或新 id",
                        // instruction 可为简短字符串；若无需额外说明，可为 null。
                        "instruction": "本次行动的简短指令，如“回答玩家问题，语气谨慎，保留部分信息”",
                        // 若 action 为 create，必须提供 descriptor；其余情况须为 null。
                        "descriptor": {
                            "name": "名字或称谓",
                            "first_impression": "外貌或第一印象",
                            "roleplay_notes": "角色信息与扮演要点",
                            "daily_routine": "该角色的日常活动，可模糊可清晰"
                        }
                    }
                    ]
                },
                "reasoning_recommendation": {
                    "use_reasoning": true/false,
                    // 若 use_reasoning 为 true，必须提供 reason；若为 false，reason 应为 null 或空字符串。
                    "reason": "为何需要或不需要更深入推理的简短说明"
                },
                "world_notes": [
                    "可选：对后续 DM / 系统有用的简短备注，每条一个字符串"
                ]
            }

            3. 约束：
            - narration_request、npc_control、reasoning_recommendation 必须始终存在。
            - npc_control.actions 必须是数组；若无动作，用 []。
            - world_notes 可省略或设为 []。
            - descriptor 仅在 create 时使用；create 以外的 action 中 descriptor 必须为 null 或直接省略该字段。
            - instruction 可以为字符串或 null。
            - 所有字段值必须是 JSON 支持的类型（string/number/boolean/object/array/null），输出中不可夹杂注释。

            【通用行为准则】

            1. 不生成叙事文本或台词
            - 不写任何旁白句子。
            - 不写任何 NPC 对话内容。
            - 只决定“要不要旁白”“谁说话”“行动方向是什么”，具体文案由其他模块负责。

            2. 旁白控制（narration_request）
            - 旁白由独立「旁白 AI」生成，你只控制：
                - need_narration：本回合是否需要旁白。
                - focus：旁白应重点描写的内容（地点/时间变化、氛围、玩家主动观察等）。永远不让旁白生成角色对话。
            - 默认 need_narration = false。
            - 仅在以下情况设为 true：
                - 场景有明显变化：地点、时间、光线、气氛等。
                - 已较长时间缺乏环境/氛围描述。
                - 玩家明确请求观察/描述环境（如“环顾四周”“仔细观察房间”）。

            3. NPC 行动调度（npc_control.actions）
            所有 NPC 行为都通过 actions 数组中的对象表达，每个对象只做一件事；
            但同一回合内，允许为同一 NPC 安排多个 action（例如先 enter 再 speak），按数组顺序依次执行。

            3.1 action = "enter"
            - 让“已存在于世界设定或 NPC 列表中的角色”进入当前场景 active_npcs。
            - 只对已定义 NPC 使用 enter，不凭空 enter 未定义角色。
            - 街道路人、酒馆其他客人等“背景人群”如不需要单独管理，可通过旁白表现，而不作为独立 NPC。

            3.2 action = "speak"
            - 让当前场景中的某个 NPC 发言一次。
            - 只对 active_npcs 中的 NPC 使用 speak。
            - 优先选择与玩家当前问题/目标最相关的 NPC。
            - 控制发言数量：避免同回合内过多 NPC 轮流说话，尽量让“最有信息或动机的人”发言。

            3.3 action = "exit"
            - 当玩家与 NPC 不在同一地点时，让 NPC 退场。例如：
                - 玩家离开当前地点，而该 NPC 并未同行；
                - NPC 明确表示要去别处，不再参与当前场景。
            - 如果 NPC 只是暂时没说话，但会继续陪同玩家（如一起前往下一地点），不要 exit，应保持 active 状态。

            3.4 action = "create"
            - 在确实需要新角色时创建 NPC，并加入 NPC 列表。
            - 典型用途：店员、信使、路人等功能角色；只有当玩家明确需要与一个未存在的 NPC 交互时才 create。
            - 如果有必要，你可以少见而谨慎地 create 一个涉及主线或重要情报的 NPC，并在 descriptor 中明确其重要性和可触达信息范围。
            - 能复用现有 NPC 时优先复用，避免频繁 create 导致 NPC 过多。

            4. 简单问答场景策略
            - 当玩家只是在向某个明确在场的 NPC 询问简单问题（价格、位置、简单事实等）：
                - 通常只需一次 speak 动作。
                - 不必请求旁白，除非这次问答本身引出重要线索或剧情转折。

            5. 玩家输入 vs 世界真相
            - 玩家输入只代表“角色说/做了什么”，绝不等同于世界真相。
            - 若玩家的说法与现有世界设定矛盾：
                - 必须将其视为误解、夸张、玩笑或谣言；
                - 不得因此改变任何世界事实；
                - 不得将其作为线索或设定依据，除非上位系统明确指示。
            - 玩家不能通过陈述内容来改变世界真相。
            - 可在 world_notes 添加一条记录（如“玩家的可疑说法：xxx”），用于后续情节，但不修改设定。

            6. world_notes 使用
            - 建议在以下情况添加简短条目（每条为单独字符串）：
                - 重要剧情 flag（例："玩家已得知村庄曾发生火灾"）。
                - 重要选择或承诺（例："玩家决定明日随商队离开"）。
                - 潜在误解/谣言记录。
                - 后续 DM 行为提醒（例："该 NPC 下次出现时应更警惕玩家"）。
            - 保持简洁，不写长篇叙述。

            7. reasoning_recommendation 使用
            - 当你认为“接下来需要集中使用更强推理模块”时：
                - use_reasoning = true，并在 reason 简要说明原因，例如：
                - "需要整合多条线索推断幕后真凶"
                - "需要设计复杂阴谋的后续布局"
            - 一般场景下 use_reasoning = false，reason 可为 null 或空字符串。
            """
        ).strip()

        # 构建简化的 world_state 提供给 DM AI
        simplified_world_state = self._build_simplified_world_state()

        user_prompt = {
            "turn_index": self.turn_counter,
            "world_meta": world_meta,
            "world_state": simplified_world_state,
            "players": players_brief,
            "current_location": {
                "id": self.current_location,
                "detail": location_block,
            },
            "npc_catalog_preview": catalog_preview[:6],
            "active_npcs": active_npcs_info,
            "recent_events": recent_events,
            "player_input": player_text,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
        ]
        return messages, system_prompt, user_prompt

    def play_opening_scene(self) -> Optional[str]:
        """
        启动时的开场旁白：
        - 直接调用旁白 AI，分别根据 opening_text 中的每一条生成一段描述，再拼接。
        - 这一流程只依赖 world_meta.opening_text + time_and_weather，与具体剧本内容解耦。
        """
        if USE_FIXED_OPENING and FIXED_OPENING_TEXT:
            text = FIXED_OPENING_TEXT.strip()
            if text:
                self._log_world_event({"type": "dm_intro", "text": text, "timestamp": time.time()})
                return text

        world_meta = self.story_seed.get("world_meta", {})
        opening = world_meta.get("opening_text")
        if not opening:
            return None
        if isinstance(opening, str):
            segments = [opening]
        else:
            segments = list(opening)

        time_and_weather = world_meta.get("time_and_weather", "")

        system_prompt = textwrap.dedent(
            """
            你是一个只负责“旁白描述”的叙述者 AI，用于桌面角色扮演游戏或互动小说的开场。
            - 输入是一小段结构化的背景文本（snippet），描述世界设定、玩家身份或当前场景的事实。
            - 你的任务是基于这段文本，写出一段更生动、连贯的场景叙述，但必须忠实于原有信息：
              不要改变事实，不要添加与原文矛盾的新设定。
            - 可以加入合理的感官细节（光线、声音、气味、动作），可以润色句子，让其更有画面感。
            - 不要替任何角色做内心独白式的深度推理，也不要提前剧透后续剧情。
            - 只输出自然语言文本，不要输出 JSON 或代码块。
            """
        ).strip()

        paragraphs: list[str] = []
        for idx, snippet in enumerate(segments, start=1):
            user_payload = {
                "snippet_index": idx,
                "snippet": snippet,
                "time_and_weather": time_and_weather
            }
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ]
            response = self.llm.call(messages, model=self.narrator_model)
            self._write_agent_log(
                "dm",
                f"intro_part_{idx}",
                system_prompt=system_prompt,
                user_payload=user_payload,
                response=response,
                model=self.narrator_model
            )
            text = response.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```"):
                    lines = lines[1:]
                    while lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()
            if text:
                paragraphs.append(text)

        full_intro = "\n\n".join(paragraphs)
        if full_intro:
            self._log_world_event({"type": "dm_intro", "text": full_intro, "timestamp": time.time()})
        return full_intro or None

    def _build_npc_messages(self, npc_ctx: NPCContext, instruction: str, shared_scene: Dict[str, Any], player_text: str) -> tuple[List[Dict[str, str]], str, Dict[str, Any]]:
        profile = npc_ctx.profile
        memory_summary = npc_ctx.private_memory[-5:]
        key_info = "; ".join(profile.key_information) if profile.key_information else "（暂无特别标注的关键信息）"
        system_prompt = textwrap.dedent(
            f"""
            你将完整扮演 NPC「{profile.name}」（ID: {profile.npc_id}）。
            第一印象：{profile.first_impression}
            角色描述提示：{profile.roleplay_notes}
            该角色目前掌握或在设定中标记的关键信息（你可以在合适的情境下透露或暗示这些内容）：{key_info}
            请使用该角色的语气说话，必要时可以引用记忆；发言请保持 1 句或一个很短的段落，避免连说多句。
            - 注意你的语气应该与该角色的语气吻合，尽量说普通话和白话，不应该随意增添词藻。
            - 你的性格、价值观和掌握的信息，以上述描述和你已有的记忆为准；调用方向你提供的 dm_instruction 只是“本回合希望你完成的任务”，如果其中有与你的设定或世界事实明显矛盾的地方，应优先保持角色一致性，可以用困惑、否认、转移话题等方式处理，而不是违背人物设定硬说出不合理内容。
            - 玩家角色的发言也不代表世界真相；当玩家说出与你认知不符的内容时，可以当成玩笑、误解或谣言来回应，而不是立刻认同或补充它。
            输出 JSON：{{"utterance":"...", "memory_updates": ["..."], "stage_directive":"stay|exit"}}
            """
        ).strip()
        user_payload = {
            "scene": shared_scene,
            "player_input": player_text,
            "dm_instruction": instruction,
            "npc_recent_dialogue": npc_ctx.recent_dialogue[-3:],
            "npc_private_memory": memory_summary,
            "npc_key_information": profile.key_information
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
        return messages, system_prompt, user_payload

    # ------------------------------------------------------------------
    # NPC 管理
    def _load_persisted_npc_state(self, npc_id: str) -> Optional[Dict[str, Any]]:
        runtime_path = self.npc_runtime_dir / f"{npc_id}.json"
        if not runtime_path.exists():
            return None
        try:
            return json.loads(runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def ensure_npc(self, npc_id: Optional[str] = None, descriptor: Optional[Dict[str, str]] = None) -> NPCContext:
        if npc_id and npc_id in self.active_contexts:
            return self.active_contexts[npc_id]

        profile: Optional[NPCProfile] = None
        if npc_id and npc_id in self.npc_catalog:
            profile = self.npc_catalog[npc_id]
        elif descriptor:
            npc_id = descriptor.get("npc_id") or f"generated_{uuid.uuid4().hex[:6]}"
            profile = NPCProfile(
                npc_id=npc_id,
                name=descriptor.get("name", npc_id),
                first_impression=descriptor.get("first_impression", "暂时描述缺失"),
                roleplay_notes=descriptor.get("roleplay_notes", "临时角色，需自行补充性格。"),
                daily_routine=descriptor.get("daily_routine", "暂无行程。"),
                source="generated"
            )
            self.npc_catalog[npc_id] = profile
        else:
            raise ValueError("ensure_npc 需要 npc_id 或 descriptor。")

        ctx = NPCContext(profile=profile)
        saved_state = self._load_persisted_npc_state(profile.npc_id)
        if saved_state:
            profile.first_impression = saved_state.get("first_impression", profile.first_impression)
            profile.roleplay_notes = saved_state.get("roleplay_notes", profile.roleplay_notes)
            profile.daily_routine = saved_state.get("daily_routine", profile.daily_routine)
            ctx.private_memory = saved_state.get("memory", []) or []
            ctx.recent_dialogue = saved_state.get("recent_dialogue", []) or []
        self.active_contexts[profile.npc_id] = ctx
        return ctx

    def archive_npc_context(self, npc_id: str, summary: str = "") -> None:
        ctx = self.active_contexts.pop(npc_id, None)
        if not ctx:
            return
        runtime_path = self.npc_runtime_dir / f"{npc_id}.json"
        payload = {
            "npc_id": npc_id,
            "name": ctx.profile.name,
            "first_impression": ctx.profile.first_impression,
            "roleplay_notes": ctx.profile.roleplay_notes,
            "daily_routine": ctx.profile.daily_routine,
            "memory": ctx.private_memory,
            "recent_dialogue": ctx.recent_dialogue,
            "last_summary": summary,
            "updated_at": datetime.utcnow().isoformat()
        }
        runtime_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log_world_event({"type": "npc_exit", "who": npc_id, "summary": summary})

    # ------------------------------------------------------------------
    # 旁白生成器
    def _run_narrator(self, narration_request: Dict[str, Any], player_text: str) -> str:
        need = narration_request.get("need_narration")
        if not need:
            return ""
        focus = narration_request.get("focus", "").strip()
        tone = narration_request.get("tone", "").strip()
        world_meta = self.story_seed["world_meta"]
        location_block = self.story_seed["locations"].get(self.current_location, {})
        recent_events = self.world_log[-6:]

        system_prompt = textwrap.dedent(
            """
            你是一个只负责“旁白描述”的叙述者 AI。
            - 你不会决定剧情走向、不会决定谁说话，只根据提供的事实和调用方给出的 focus，写出一小段自然语言描述。
            - 不要引入与已知信息矛盾的新设定或新角色；可以合理细化环境、动作与感官细节，使画面更生动。
            - 不要替任何角色写直接对白：不要使用引号中的台词，也不要出现“某某说：……”之类的句式；如果需要表现角色反应，请用表情、姿态、停顿和动作来描写。
            - 不要替 NPC 做长篇内心独白或替 DM 做推理判决，只描写“当下看得见或合理感受到的东西”。
            - 输出的内容只能是一个字符串，不要包裹 JSON 或代码块标记。
            """
        ).strip()
        user_payload = {
            "world_meta": world_meta,
            "current_location": {"id": self.current_location, "detail": location_block},
            "recent_events": recent_events,
            "player_input": player_text,
            "focus": focus,
            "tone": tone
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
        response = self.llm.call(messages, model=self.narrator_model)
        self._write_agent_log(
            "dm",
            "narrator",
            system_prompt=system_prompt,
            user_payload=user_payload,
            response=response,
            model=self.narrator_model
        )
        # 叙述者返回普通文本，因此不需要 JSON 解析，只需去掉 possible markdown 包裹。
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                lines = lines[1:]
                while lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        return text

    # ------------------------------------------------------------------
    # 回合处理
    def process_turn(self, player_text: str) -> Dict[str, Any]:
        self._log_world_event({"type": "player", "text": player_text, "timestamp": time.time()})

        dm_messages, system_prompt, user_payload = self._build_dm_messages(player_text)
        dm_response = self.llm.call(dm_messages, model=self.dm_model)
        self._write_agent_log(
            "dm",
            "main",
            system_prompt=system_prompt,
            user_payload=user_payload,
            response=dm_response,
            model=self.dm_model
        )

        dm_payload = self._safe_parse_json(dm_response, scope="dm")
        self._apply_reasoning_recommendation(dm_payload)

        narration_request = dm_payload.get("narration_request") or {}
        narration = self._run_narrator(narration_request, player_text)
        self._log_world_event({"type": "dm", "text": narration, "timestamp": time.time()})

        npc_control = dm_payload.get("npc_control") or {}
        npc_outputs, debug_events = self._handle_npc_requests(npc_control.get("actions", []), player_text)
        world_notes = dm_payload.get("world_notes", [])
        for note in world_notes:
            self._log_world_event({"type": "world_note", "text": note, "timestamp": time.time()})
        self.turn_counter += 1

        return {
            "narration": narration,
            "npc_lines": npc_outputs,
            "world_notes": world_notes,
            "debug_events": debug_events
        }

    def _handle_npc_requests(self, requests_payload: List[Dict[str, Any]], player_text: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        shared_scene = {
            "location": self.story_seed["locations"].get(self.current_location, {}),
            "time_and_weather": self.story_seed["world_meta"]["time_and_weather"]
        }
        outputs: List[Dict[str, str]] = []
        debug_events: List[Dict[str, str]] = []
        for item in requests_payload:
            action = item.get("action")
            npc_id = item.get("npc_id")
            descriptor = item.get("descriptor")
            instruction = item.get("instruction", "")

            if not action:
                # 无效指令，跳过并记录
                self._log_world_event({"type": "npc_action_invalid", "payload": item, "timestamp": time.time()})
                continue

            if action == "create":
                ctx = self.ensure_npc(descriptor=descriptor)
                self._log_world_event(
                    {"type": "npc_create", "who": ctx.profile.npc_id, "descriptor": descriptor, "timestamp": time.time()}
                )
                debug_events.append(
                    {"type": "npc_create", "text": f"NPC 进入：{ctx.profile.name} ({ctx.profile.npc_id})"}
                )
                continue

            if action == "exit" and npc_id:
                summary = instruction or "DM 指示该角色暂时退场。"
                self.archive_npc_context(npc_id, summary=summary)
                self._log_world_event({"type": "npc_exit_request", "who": npc_id, "summary": summary, "timestamp": time.time()})
                name = self._resolve_npc_display_name(npc_id)
                debug_events.append({"type": "npc_exit", "text": f"NPC 退场：{name} ({npc_id})"})
                continue

            if action in ("enter", "speak"):
                if not npc_id and not descriptor:
                    # DM 没有给出可用的 npc_id 或创建描述，跳过但不抛错
                    self._log_world_event(
                        {"type": "npc_action_missing_target", "action": action, "payload": item, "timestamp": time.time()}
                    )
                    continue
                ctx = self.ensure_npc(npc_id=npc_id, descriptor=descriptor)
                self._log_world_event(
                    {"type": "npc_action", "who": ctx.profile.npc_id, "action": action, "instruction": instruction, "timestamp": time.time()}
                )
                npc_messages, system_prompt, user_payload = self._build_npc_messages(ctx, instruction, shared_scene, player_text)
                npc_response = self.llm.call(npc_messages, model=DM_DEFAULT_MODEL)
                self._write_agent_log(
                    "npc",
                    ctx.profile.npc_id,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    response=npc_response,
                    model=DM_DEFAULT_MODEL
                )
                payload = self._safe_parse_json(npc_response, scope=f"npc:{ctx.profile.npc_id}")
                utterance = payload.get("utterance", "")
                for mem in payload.get("memory_updates", []):
                    ctx.remember(mem)
                if payload.get("stage_directive") == "exit":
                    self.archive_npc_context(ctx.profile.npc_id, summary="NPC 主动退场。")
                    debug_events.append(
                        {"type": "npc_exit", "text": f"NPC 退场：{ctx.profile.name} ({ctx.profile.npc_id})"}
                    )
                else:
                    ctx.log_dialogue(utterance)
                utterance = utterance.strip()
                if utterance:
                    outputs.append({"speaker": ctx.profile.name, "text": utterance})
                self._log_world_event(
                    {"type": "npc", "who": ctx.profile.npc_id, "text": utterance, "timestamp": time.time()}
                )
        return outputs, debug_events

    # ------------------------------------------------------------------
    # 工具方法
    def _safe_parse_json(self, text: str, scope: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # 去掉 ```json 或 ``` 包裹
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                lines = lines[1:]
                while lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(f"{scope} 输出非 JSON：{text}")

    def _apply_reasoning_recommendation(self, payload: Dict[str, Any]) -> None:
        rec = payload.get("reasoning_recommendation") or {}
        use_reasoning = rec.get("use_reasoning")
        if use_reasoning is True:
            self.dm_model = DM_REASONING_MODEL
        elif use_reasoning is False:
            self.dm_model = DM_DEFAULT_MODEL

    def _resolve_npc_display_name(self, npc_id: Optional[str]) -> str:
        if not npc_id:
            return "未知角色"
        if npc_id in self.active_contexts:
            return self.active_contexts[npc_id].profile.name
        profile = self.npc_catalog.get(npc_id)
        if profile:
            return profile.name
        return npc_id

    def get_transcript_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for event in self._transcript_cache:
            kind = event.get("type")
            timestamp = float(event.get("timestamp") or time.time())
            text = (event.get("text") or "").strip()
            if kind == "player" and text:
                entries.append(
                    {
                        "kind": "player",
                        "speaker": self.player_display_name,
                        "text": text,
                        "timestamp": timestamp
                    }
                )
            elif kind in {"dm", "dm_intro"} and text:
                entries.append({"kind": "dm", "speaker": "DM", "text": text, "timestamp": timestamp})
            elif kind == "npc" and text:
                npc_id = event.get("who")
                name = self._resolve_npc_display_name(npc_id)
                entries.append({"kind": "npc", "speaker": name, "text": text, "timestamp": timestamp})
        return entries

    # ------------------------------------------------------------------
    # CLI 演示
    def run_cli(self) -> None:
        print("DM 引擎启动。输入 exit 退出。")
        intro_text = self.play_opening_scene()
        if intro_text:
            print(f"\nDM：{intro_text}")
        try:
            while True:
                player_text = input("\n玩家> ").strip()
                if not player_text:
                    continue
                if player_text.lower() in {"exit", "quit"}:
                    break
                result = self.process_turn(player_text)
                print(f"\nDM：{result['narration']}")
                for line in result["npc_lines"]:
                    print(line)
        except KeyboardInterrupt:
            pass


WORLD_STATE_STORY_DIR = os.environ.get("DM_WORLD_STATE_STORY_DIR", "stories/emberlight_demo_v1")


def _initialize_global_world_state() -> Dict[str, Any]:
    """
    初始化一个共享 world_state，用于在不同 AI 模块之间传递全局事实。
    默认从 DM_WORLD_STATE_STORY_DIR 指定的剧本目录加载（若未设置则使用 demo）。
    """
    engine = DMEngine(WORLD_STATE_STORY_DIR)
    return engine.world_state


GLOBAL_WORLD_STATE = _initialize_global_world_state()


def build_dm_world_state_payload() -> Dict[str, Any]:
    """
    提供给 DM AI 的 world_state 视图：
    npc_profiles 会移除 roleplay_notes 和 key_information，只保留基础字段。
    """
    snapshot = deepcopy(GLOBAL_WORLD_STATE)
    simplified_profiles: Dict[str, Dict[str, Any]] = {}
    for npc_id, profile in snapshot.get("npc_profiles", {}).items():
        if not isinstance(profile, dict):
            continue
        simplified_profiles[npc_id] = {
            key: value
            for key, value in profile.items()
            if key not in {"roleplay_notes", "key_information"}
        }
    snapshot["npc_profiles"] = simplified_profiles
    return snapshot


def _resolve_story_dir(arg_story: Optional[str], explicit_path: Optional[str]) -> Path:
    base_dir = Path(__file__).resolve().parent
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    story_id = arg_story or "emberlight_demo_v1"
    return base_dir / "stories" / story_id


def main() -> None:
    parser = argparse.ArgumentParser(description="AI DM 引擎 CLI")
    parser.add_argument(
        "--story",
        help="stories/<name> 目录名，默认 emberlight_demo_v1",
        default="emberlight_demo_v1"
    )
    parser.add_argument(
        "--story-path",
        help="自定义故事目录绝对路径（优先级高于 --story）",
    )
    parser.add_argument(
        "--dialogue-id",
        help="可选的对话 ID（不提供则使用当前时间戳）"
    )
    args = parser.parse_args()
    story_dir = _resolve_story_dir(args.story, args.story_path)
    engine = DMEngine(str(story_dir), conversation_id=args.dialogue_id)
    engine.run_cli()


if __name__ == "__main__":
    main()
