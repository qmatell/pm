#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw Manager
v2.0.2
修改备份位置，优化删除 __pycache__逻辑
功能：
1. 检测 API 是否可用（check）
2. 添加 provider 并自动注册该 provider 的全部模型（add）
3. 删除 provider（支持批量/序号/空格分隔）并清理已注册模型（remove）
4. 同步 provider 的模型到 agents.defaults.models（sync，可选单个/多选，空格分隔）
5. 列出当前 provider（list，带序号），以及单个 provider 模型列表（models）
6. 快速切换默认模型（switch：按 provider+模型序号/名称）
7. 安装/自定义别名（alias-install，默认别名 opm）

示例：
  python3 provider_manager.py check --base-url https://api.xxx.com/v1 --api-key sk-xxx
  python3 provider_manager.py add --name myp --base-url https://api.xxx.com/v1 --api-key sk-xxx
  python3 provider_manager.py remove --names 1 3 myp
  python3 provider_manager.py sync --providers 1 myp
  python3 provider_manager.py models --provider 3
  python3 provider_manager.py switch --provider 3 --model 2
  python3 provider_manager.py alias-install --name opm
"""

import argparse
import getpass
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import threading
import re
import unicodedata
import urllib.request
import urllib.error
import hashlib
from typing import Dict, List, Tuple

CONFIG_PATH = "/root/.openclaw/openclaw.json"
DEFAULT_UA = "curl/8.5.0"
THIS_SCRIPT = os.path.abspath(__file__)
MENU_SEP = "=" * 33
SELF_REPO_URL = "https://github.com/qmatell/opm.git"
SELF_RAW_URL = "https://raw.githubusercontent.com/qmatell/opm/main/opm.py"
SELF_COMMIT_API_URL = "https://api.github.com/repos/qmatell/opm/commits/main"
MENU_STATUS = {
    "openclaw_ver": "检测中...",
    "latest_ver": "检测中...",
    "runtime": "检测中...",
}
_MENU_PROBE_STARTED = False

INSTALL_STATUS_CACHE = {
    "ts": 0.0,
    "openclaw_ver": "",
    "latest_ver": "",
    "latest_ts": 0.0,
    "runtime": "未知",
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_cmd(cmd, check=True, timeout=None):
    """Run command without shell to avoid quoting issues."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if hasattr(e, "stdout") else ""
        if check:
            raise RuntimeError(
                f"命令超时: {' '.join(shlex.quote(x) for x in cmd)}\n{out}"
            )
        return 124, out

    if check and p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(shlex.quote(x) for x in cmd)}\n{p.stdout}")
    return p.returncode, p.stdout


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_base_url(url: str) -> str:
    u = url.strip()
    while u.endswith("/"):
        u = u[:-1]
    if not (u.startswith("http://") or u.startswith("https://")):
        raise ValueError("base-url 必须以 http:// 或 https:// 开头")
    return u


def split_tokens(raw: str) -> List[str]:
    # 支持空格/逗号/分号混用，去重保持顺序
    if not raw:
        return []
    for ch in [",", ";"]:
        raw = raw.replace(ch, " ")
    toks = [t for t in raw.split() if t.strip()]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_models(base_url: str, api_key: str, timeout: int = 10, ua: str = DEFAULT_UA) -> Tuple[bool, int, List[str], str]:
    """Call GET {base_url}/models with Bearer token and parse model ids."""
    url = f"{normalize_base_url(base_url)}/models"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", ua)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return False, e.code, [], body
    except Exception as e:
        return False, None, [], str(e)

    try:
        data = json.loads(body)
    except Exception:
        return False, status, [], f"返回非 JSON：{body[:500]}"

    models = []
    items = data.get("data", []) if isinstance(data, dict) else []
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))

    # 去重保持顺序
    seen = set()
    model_ids = []
    for m in models:
        if m not in seen:
            seen.add(m)
            model_ids.append(m)

    ok = status == 200 and len(model_ids) > 0
    if ok:
        return True, status, model_ids, body
    return False, status, model_ids, body


def build_provider_models(provider_name, model_ids, inputs, context_window, max_tokens):
    models = []
    for mid in model_ids:
        models.append(
            {
                "id": mid,
                "name": f"{provider_name} / {mid}",
                "input": inputs,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
                "contextWindow": context_window,
                "maxTokens": max_tokens,
            }
        )
    return models


def probe_chat_model(base_url: str, api_key: str, model_id: str, timeout: int = 12, ua: str = DEFAULT_UA) -> Tuple[bool, int, str]:
    """Probe chat/completions for one model; return (ok, status, detail)."""
    url = f"{normalize_base_url(base_url)}/chat/completions"
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
    }
    req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode("utf-8"))
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", ua)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
            return resp.getcode() == 200, resp.getcode(), txt[:200]
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return False, e.code, txt[:200]
    except Exception as e:
        return False, 0, str(e)


def filter_models_by_chat(
    base_url: str,
    api_key: str,
    model_ids: List[str],
    timeout: int,
    ua: str,
    max_probe: int = 0,
    progress_prefix: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Validate model ids by probing /chat/completions.
    max_probe=0 means probe all models.
    Returns (ok_models, bad_models).
    """
    ids = model_ids[:]
    if max_probe and max_probe > 0:
        ids = ids[:max_probe]

    ok, bad = [], []
    total = len(ids)
    prefix = (progress_prefix or "").strip()
    if prefix:
        prefix = prefix + " "

    for i, mid in enumerate(ids, 1):
        succ, code, _ = probe_chat_model(base_url, api_key, mid, timeout=timeout, ua=ua)
        if succ:
            ok.append(mid)
            print(f"{prefix}{i}/{total} ✅ {mid}")
        else:
            bad.append(mid)
            c = f" HTTP:{code}" if code else ""
            print(f"{prefix}{i}/{total} ❌ {mid}{c}")

    # if limited probe, keep unprobed models as-is (append after validated ok)
    if max_probe and max_probe > 0 and len(model_ids) > max_probe:
        remaining = model_ids[max_probe:]
        for mid in remaining:
            print(f"{prefix}➖ 未检测(按可用保留): {mid}")
        ok.extend(remaining)

    return ok, bad


def config_set(path_key: str, obj):
    payload = json.dumps(obj, ensure_ascii=False)
    run_cmd(["openclaw", "config", "set", path_key, "--json", payload], check=True)


def restart_gateway():
    # 部分环境（无 systemd user bus）下，openclaw gateway restart 会较慢并失败
    # 这里加短超时并优先判断是否已可连通，减少卡顿与重复报错
    rc, out = run_cmd(["openclaw", "gateway", "restart"], check=False, timeout=8)
    if rc == 0:
        print("✅ gateway 已重启（openclaw gateway restart）")
        return True

    # 如果 restart 失败但 gateway RPC 探测正常，视为可用，不再兜底重启
    try:
        rc_st, out_st = run_cmd(["openclaw", "gateway", "status"], check=False, timeout=8)
        txt = (out_st or "").lower()
        if (rc_st == 0) and (("rpc probe: ok" in txt) or ("probe: ok" in txt)):
            print("✅ gateway 可用（检测到 RPC probe: ok，跳过兜底重启）")
            return True
    except Exception:
        pass

    eprint("⚠️ openclaw gateway restart 失败，尝试兜底重启...")
    try:
        rc, out = run_cmd(["pgrep", "-f", "openclaw-gateway"], check=False, timeout=5)
        if rc == 0 and out.strip():
            for pid_s in out.strip().splitlines():
                try:
                    os.kill(int(pid_s.strip()), signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(1)
        print("✅ 已发送 gateway 重启信号（兜底方式）")
        return True
    except Exception as e:
        eprint(f"❌ 兜底重启也失败：{e}")
        return False


def get_provider_list(cfg: Dict) -> List[str]:
    providers = cfg.get("models", {}).get("providers", {})
    if not isinstance(providers, dict):
        return []
    return sorted(providers.keys())


def parse_selection(sel: str, names: List[str]) -> List[str]:
    """
    支持输入："1 3 myprovider"，数字按 list 序号（从1开始），其余按名字。
    为空则返回空列表（表示默认行为由调用方决定）。
    """
    tokens = split_tokens(sel)
    if not tokens:
        return []
    selected = []
    for t in tokens:
        if t.isdigit():
            idx = int(t)
            if 1 <= idx <= len(names):
                selected.append(names[idx - 1])
        else:
            if t in names:
                selected.append(t)
    # 去重保持顺序
    seen = set()
    out = []
    for n in selected:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def cmd_check(args):
    base_url = (getattr(args, "base_url", "") or "").strip()
    api_key = (getattr(args, "api_key", "") or "").strip()
    ua = (getattr(args, "ua", DEFAULT_UA) or DEFAULT_UA).strip() or DEFAULT_UA
    timeout = int(getattr(args, "timeout", 10) or 10)

    providers_raw = getattr(args, "providers", "0")
    if isinstance(providers_raw, list):
        providers_sel = " ".join(str(x) for x in providers_raw if str(x).strip())
    else:
        providers_sel = str(providers_raw or "0")
    providers_sel = providers_sel.strip()

    new_api_mode = bool(getattr(args, "new_api", False))

    def check_one(base_url_x: str, api_key_x: str) -> int:
        ok, status, model_ids, detail = fetch_models(base_url_x, api_key_x, timeout, ua)
        if ok:
            print("✅ API 可用")
            print(f"HTTP: {status}")
            print(f"模型数: {len(model_ids)}")
            for i, m in enumerate(model_ids, 1):
                print(f"{i}. {m}")
            return 0
        print("❌ API 不可用")
        print(f"HTTP: {status}")
        print(f"详情: {detail[:800]}")
        return 2

    # 新 API 检测模式
    if new_api_mode:
        if not base_url:
            base_url = input("请输入 base-url (例如 https://api.xxx.com/v1，输入q取消): ").strip()
            if base_url.lower() == "q":
                print("已取消")
                return 0
        if not api_key:
            api_key = getpass.getpass("请输入 api-key（输入q取消）: ").strip()
            if api_key.lower() == "q":
                print("已取消")
                return 0
        return check_one(base_url, api_key)

    # 兼容：若传了 base-url/api-key 也按单 API 检测
    if base_url or api_key:
        if not base_url:
            base_url = input("请输入 base-url (例如 https://api.xxx.com/v1，输入q取消): ").strip()
            if base_url.lower() == "q":
                print("已取消")
                return 0
        if not api_key:
            api_key = getpass.getpass("请输入 api-key（输入q取消）: ").strip()
            if api_key.lower() == "q":
                print("已取消")
                return 0
        return check_one(base_url, api_key)

    # 默认检测本地已添加 provider 的 API
    cfg = load_config()
    names_sorted = get_provider_list(cfg)
    providers = cfg.get("models", {}).get("providers", {})

    if not names_sorted:
        print("❌ 未找到已添加的 provider")
        return 1

    if providers_sel in ("", "0"):
        targets = names_sorted
    else:
        targets = parse_selection(providers_sel, names_sorted)
        if not targets:
            print("❌ 未匹配到有效 provider")
            return 1

    print(f"检测目标 provider 数: {len(targets)}")
    rc_final = 0
    ok_count = 0
    bad_count = 0
    bad_indices = []
    for idx, pname in enumerate(targets, 1):
        pobj = providers.get(pname) or {}
        p_base = str(pobj.get("baseUrl") or "").strip()
        p_key = str(pobj.get("apiKey") or "").strip()

        if (not p_base) or (not p_key):
            print(f"{idx}. ❌ {pname}  配置缺失(baseUrl/apiKey)")
            rc_final = 2
            bad_count += 1
            bad_indices.append(str(idx))
            continue

        ok, status, model_ids, detail = fetch_models(p_base, p_key, timeout, ua)
        if ok:
            print(f"{idx}. ✅ {pname}  HTTP:{status}  模型数:{len(model_ids)}")
            ok_count += 1
        else:
            print(f"{idx}. ❌ {pname}  HTTP:{status}  详情:{(detail or '')[:120]}")
            rc_final = 2
            bad_count += 1
            bad_indices.append(str(idx))

    print(MENU_SEP)
    print(f"检测结果：可用 {ok_count} 个，不可用 {bad_count} 个")
    if bad_indices:
        print(f"不可用序号：{' '.join(bad_indices)}")

    return rc_final


def cmd_add(args):
    provider = (args.name or "").strip()
    if not provider:
        provider = input("请输入 provider 名称 (例如 myprovider，输入q取消): ").strip()
    if provider.lower() == "q":
        print("已取消")
        return 0
    if not provider:
        raise ValueError("provider name 不能为空")

    base_url = (args.base_url or "").strip()
    if not base_url:
        base_url = input("请输入 base-url (例如 https://api.xxx.com/v1，输入q取消): ").strip()
    if base_url.lower() == "q":
        print("已取消")
        return 0
    base_url = normalize_base_url(base_url)

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = getpass.getpass("请输入 api-key（输入q取消）: ").strip()
    if api_key.lower() == "q":
        print("已取消")
        return 0

    ua = args.ua.strip() or DEFAULT_UA

    ok, status, model_ids, detail = fetch_models(base_url, api_key, args.timeout, ua)
    if not ok:
        print("❌ 先检测 API 失败，已取消添加")
        print(f"HTTP: {status}")
        print(f"详情: {detail[:800]}")
        return 2

    # 始终仅保留 chat/completions 实测可用模型，避免“能列出但调用404”
    model_ids, bad_models = filter_models_by_chat(
        base_url,
        api_key,
        model_ids,
        timeout=args.timeout,
        ua=ua,
        max_probe=args.max_probe,
        progress_prefix=f"[{provider}]",
    )
    if bad_models:
        print(f"⚠️ 已过滤 chat 不可用模型: {len(bad_models)}")
    if not model_ids:
        print("❌ 过滤后没有可用模型，取消添加")
        return 2

    inputs = [x.strip() for x in split_tokens(args.inputs) if x.strip()]
    if not inputs:
        inputs = ["text", "image"]

    provider_obj = {
        "baseUrl": base_url,
        "apiKey": api_key,
        "api": args.api,
        "models": build_provider_models(
            provider, model_ids, inputs, args.context_window, args.max_tokens
        ),
    }

    print(f"➡️ 添加 provider: {provider}")
    config_set(f"models.providers.{provider}", provider_obj)

    cfg = load_config()
    defaults_models = cfg.get("agents", {}).get("defaults", {}).get("models", {})
    if not isinstance(defaults_models, dict):
        defaults_models = {}

    added = 0
    for mid in model_ids:
        key = f"{provider}/{mid}"
        if key not in defaults_models:
            defaults_models[key] = {}
            added += 1

    config_set("agents.defaults.models", defaults_models)

    if not args.no_restart:
        restart_gateway()

    print("✅ 添加完成")
    print(f"provider: {provider}")
    print(f"模型总数: {len(model_ids)}")
    print(f"新注册到 agents.defaults.models: {added}")
    return 0


def cmd_remove(args):
    cfg = load_config()
    names_sorted = get_provider_list(cfg)
    selected = parse_selection(args.names, names_sorted)
    if not selected and args.name:
        selected = parse_selection(args.name, names_sorted)
    if not selected:
        print("❌ 没有指定要删除的 provider")
        return 1

    providers = cfg.get("models", {}).get("providers", {})
    defaults_models = cfg.get("agents", {}).get("defaults", {}).get("models", {})
    if not isinstance(defaults_models, dict):
        defaults_models = {}

    removed = []
    for pname in selected:
        if pname in providers:
            providers.pop(pname, None)
            removed.append(pname)
        # 清理注册模型
        to_del = [k for k in list(defaults_models.keys()) if k.startswith(pname + "/")]
        for k in to_del:
            defaults_models.pop(k, None)

    config_set("models.providers", providers)
    config_set("agents.defaults.models", defaults_models)
    print(f"✅ 已删除 provider: {', '.join(removed) if removed else '无'}")

    if not args.no_restart:
        restart_gateway()

    return 0


def cmd_sync(args):
    cfg = load_config()
    names_sorted = get_provider_list(cfg)
    defaults_models = cfg.get("agents", {}).get("defaults", {}).get("models", {})
    if not isinstance(defaults_models, dict):
        defaults_models = {}

    targets = parse_selection(args.providers, names_sorted) if args.providers else names_sorted
    if not targets:
        print("❌ 未匹配到要同步的 provider")
        return 1

    added = 0
    removed = 0

    for pname in targets:
        pobj = cfg.get("models", {}).get("providers", {}).get(pname) or {}
        mids = [m.get("id") for m in (pobj.get("models", []) or []) if isinstance(m, dict) and m.get("id")]

        # 始终检测可用性：仅同步可用模型
        ok_ids, bad_ids = filter_models_by_chat(
            pobj.get("baseUrl", ""),
            pobj.get("apiKey", ""),
            mids,
            timeout=args.timeout,
            ua=args.ua,
            max_probe=args.max_probe,
            progress_prefix=f"[{pname}]",
        )

        valid_set = {f"{pname}/{x}" for x in ok_ids}

        # 找出当前 defaults 里该 provider 已注册的模型
        current_keys = [k for k in list(defaults_models.keys()) if k.startswith(pname + "/")]

        # 1) 移除不可用模型或已不在 provider 模型列表里的旧注册
        rm_this = 0
        for k in current_keys:
            if k not in valid_set:
                defaults_models.pop(k, None)
                removed += 1
                rm_this += 1

        # 2) 添加本次检测可用且未注册的模型
        add_this = 0
        for mid in ok_ids:
            key = f"{pname}/{mid}"
            if key not in defaults_models:
                defaults_models[key] = {}
                added += 1
                add_this += 1

        reg_now = len([k for k in defaults_models.keys() if k.startswith(pname + "/")])

        print(
            f"{pname}: provider模型={len(mids)}, 可用={len(ok_ids)}, 不可用={len(bad_ids)}, "
            f"新增={add_this}, 移除={rm_this}, 已注册={reg_now}"
        )
        if bad_ids:
            print(f"⚠️ {pname} 不可用模型已移除: {len(bad_ids)}")

    config_set("agents.defaults.models", defaults_models)

    if not args.no_restart:
        restart_gateway()

    print(f"✅ 同步完成：新增可用模型 {added}，移除不可用/失效模型 {removed}")
    return 0


def cmd_list(_args):
    cfg = load_config()
    providers = cfg.get("models", {}).get("providers", {})
    defaults_models = cfg.get("agents", {}).get("defaults", {}).get("models", {})
    if not isinstance(defaults_models, dict):
        defaults_models = {}

    names = get_provider_list(cfg)
    print(f"provider 数量: {len(names)}")
    for idx, pname in enumerate(names, 1):
        pobj = providers.get(pname, {}) if isinstance(providers, dict) else {}
        count = len(pobj.get("models", []) or [])
        reg = len([k for k in defaults_models.keys() if k.startswith(pname + "/")])
        print(f"{idx}. {pname}: provider模型={count}, 已注册={reg}")
    return 0


def cmd_models(args):
    cfg = load_config()
    names_sorted = get_provider_list(cfg)
    providers = cfg.get("models", {}).get("providers", {})

    if not args.provider:
        print("❌ 需要 --provider 指定（可用序号/名称，支持空格分隔多个）")
        return 1

    selected = parse_selection(args.provider, names_sorted)
    if not selected:
        # 兼容单个直接名称
        raw = [x.strip() for x in split_tokens(args.provider) if x.strip()]
        selected = [x for x in raw if x in providers]

    if not selected:
        print("❌ 未匹配到有效 provider")
        return 1

    check_avail = not bool(getattr(args, "no_check", False))
    timeout = int(getattr(args, "timeout", 10) or 10)
    ua = (getattr(args, "ua", DEFAULT_UA) or DEFAULT_UA).strip() or DEFAULT_UA
    max_probe = int(getattr(args, "max_probe", 0) or 0)

    rc_final = 0
    for idx_p, pname in enumerate(selected, 1):
        pobj = providers.get(pname)
        if not pobj:
            print(f"❌ provider 不存在: {pname}")
            rc_final = 1
            continue

        models = pobj.get("models", []) or []
        mids = []
        for m in models:
            mid = m.get("id") if isinstance(m, dict) else str(m)
            if mid:
                mids.append(mid)

        print(MENU_SEP)
        print(f"[{idx_p}/{len(selected)}] provider: {pname}")
        print(f"模型数: {len(mids)}")

        if not mids:
            print("(无模型)")
            continue

        if check_avail:
            ok_set = set()
            bad_set = set()
            probe_ids = mids if max_probe <= 0 else mids[:max_probe]
            total_probe = len(probe_ids)
            for i, mid in enumerate(probe_ids, 1):
                succ, code, _detail = probe_chat_model(
                    pobj.get("baseUrl", ""),
                    pobj.get("apiKey", ""),
                    mid,
                    timeout=timeout,
                    ua=ua,
                )
                if succ:
                    ok_set.add(mid)
                    print(f"{i}/{total_probe} ✅ {mid}")
                else:
                    bad_set.add(mid)
                    c = f" HTTP:{code}" if code else ""
                    print(f"{i}/{total_probe} ❌ {mid}{c}")

            if max_probe > 0 and len(mids) > max_probe:
                for mid in mids[max_probe:]:
                    print(f"➖ 未检测(按可用保留): {mid}")

            print(f"可用: {len(ok_set)}  不可用: {len(bad_set)}  未检测: {max(0, len(mids)-len(ok_set)-len(bad_set))}")
            if bad_set:
                rc_final = 2
        else:
            for i, mid in enumerate(mids, 1):
                print(f"{i}. {mid}")

    print(MENU_SEP)
    return rc_final


def cmd_switch(args):
    cfg = load_config()
    names_sorted = get_provider_list(cfg)
    providers = cfg.get("models", {}).get("providers", {})

    # 交互式选择 provider
    pname = ""
    if args.provider:
        sel = parse_selection(args.provider, names_sorted)
        pname = sel[0] if sel else args.provider.strip()
    while not pname:
        print("请选择 provider（序号或名称），当前列表：")
        for i, n in enumerate(names_sorted, 1):
            print(f"{i}. {n}")
        try:
            inp = input("provider（输入q取消）: ").strip()
        except EOFError:
            return 1
        if inp.lower() == "q":
            print("已取消")
            return 0
        sel = parse_selection(inp, names_sorted)
        pname = sel[0] if sel else inp
        if pname and pname not in providers:
            print("❌ provider 不存在，再试一次")
            pname = ""
    if pname not in providers:
        print(f"❌ provider 不存在: {pname}")
        return 1

    models = providers[pname].get("models", []) or []
    if not models:
        print(f"❌ provider 没有模型: {pname}")
        return 1

    # 交互式选择模型
    mid = None
    if args.model:
        if args.model.isdigit():
            idx = int(args.model)
            if 1 <= idx <= len(models):
                mobj = models[idx - 1]
                mid = mobj.get("id") if isinstance(mobj, dict) else str(mobj)
        else:
            mid = args.model.strip()
    while not mid:
        print(f"模型列表 ({pname}):")
        for i, m in enumerate(models, 1):
            mid_cur = m.get("id") if isinstance(m, dict) else str(m)
            print(f"{i}. {mid_cur}")
        try:
            inp = input("模型序号或名称（空回车默认第1个，输入q取消）: ").strip()
        except EOFError:
            return 1
        if inp.lower() == "q":
            print("已取消")
            return 0
        if not inp:
            mobj = models[0]
            mid = mobj.get("id") if isinstance(mobj, dict) else str(mobj)
            break
        if inp.isdigit():
            idx = int(inp)
            if 1 <= idx <= len(models):
                mobj = models[idx - 1]
                mid = mobj.get("id") if isinstance(mobj, dict) else str(mobj)
                break
        else:
            mid = inp
            break

    target = f"{pname}/{mid}"
    config_set("agents.defaults.model.primary", target)
    print(f"✅ 默认模型已切换为: {target}")

    # 可选：同步切换当前会话模型
    if args.session:
        try:
            rc, out = run_cmd(["openclaw", "sessions", "--json"], check=True)
            data = json.loads(out)
            sessions = data.get("sessions", []) if isinstance(data, dict) else []
            if not sessions:
                print("⚠️ 未找到会话，跳过会话切换")
            else:
                latest = max(sessions, key=lambda s: s.get("updatedAt", 0))
                sid = latest.get("sessionId")
                if not sid:
                    print("⚠️ 未找到 sessionId，跳过会话切换")
                else:
                    # 尝试切换并校验（最多 2 次）
                    ok = False
                    for attempt in (1, 2):
                        run_cmd(["openclaw", "agent", "--session-id", sid, "--message", f"/model {target}", "--timeout", "60"], check=False)
                        time.sleep(1)
                        try:
                            _, out2 = run_cmd(["openclaw", "sessions", "--json"], check=True)
                            data2 = json.loads(out2)
                            sessions2 = data2.get("sessions", []) if isinstance(data2, dict) else []
                            latest2 = max(sessions2, key=lambda s: s.get("updatedAt", 0)) if sessions2 else {}
                            cur = f"{latest2.get('modelProvider','')}/{latest2.get('model','')}".strip("/")
                            if cur == target:
                                ok = True
                                break
                        except Exception:
                            pass
                    if ok:
                        print(f"✅ 当前会话已切换为: {target}")
                    else:
                        print(f"⚠️ 会话切换未生效，当前仍为: {cur if 'cur' in locals() else 'unknown'}")
        except Exception as e:
            print(f"⚠️ 会话切换失败: {e}")

    if not args.no_restart:
        restart_gateway()
    return 0


def install_alias(alias_name: str = "opm", path_override: str = ""):
    # 尝试用户指定路径，否则 /usr/local/bin，再其次 ~/bin
    paths = []
    if path_override:
        paths.append(path_override)
    paths.extend(["/usr/local/bin", os.path.expanduser("~/bin")])
    target = os.path.realpath(THIS_SCRIPT)
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
            dst = os.path.join(p, alias_name)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(target, dst)
            os.chmod(dst, 0o755)
            print(f"✅ 已创建别名：{dst} -> {target}")
            return 0
        except PermissionError:
            continue
        except Exception as e:
            eprint(f"⚠️ 创建别名失败: {e}")
            continue
    print("❌ 无法写入目标路径；请手动创建 alias 或使用 sudo。")
    return 1


def cmd_alias(args):
    name = args.name.strip() or "opm"
    return install_alias(name, args.path.strip())


def clear_screen():
    try:
        os.system("clear" if os.name != "nt" else "cls")
    except Exception:
        pass


def pause_any_key(msg: str = "\n按任意键返回主界面..."):
    # TTY 下真正读取单键；非 TTY 回退为回车
    print(msg, end="", flush=True)
    if not sys.stdin.isatty():
        try:
            input()
        except Exception:
            pass
        return

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
    except Exception:
        try:
            input()
        except Exception:
            pass


def get_primary_model() -> str:
    try:
        cfg = load_config()
        return (
            cfg.get("agents", {})
            .get("defaults", {})
            .get("model", {})
            .get("primary", "(未设置)")
        )
    except Exception:
        return "(读取失败)"


def yes_no(prompt: str, default: bool = False) -> bool:
    tip = "Y/n" if default else "y/N"
    while True:
        try:
            s = input(f"{prompt} [{tip}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not s:
            return default
        if s in ("q", "Q"):
            return default
        if s in ("y", "yes", "1", "true"):
            return True
        if s in ("n", "no", "false", "2"):
            return False
        print("请输入 y 或 n（或直接回车）")


def to_int(raw: str, default: int) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def prompt_text(prompt: str, default: str = "", allow_zero_cancel: bool = False):
    try:
        s = input(prompt).strip()
    except EOFError:
        return None if allow_zero_cancel else default
    if allow_zero_cancel and s.lower() == "q":
        return None
    if not s:
        return default
    return s


def run_action(func, args_obj) -> int:
    try:
        return int(func(args_obj) or 0)
    except Exception as e:
        eprint(f"❌ 执行失败: {e}")
        return 1


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def openclaw_version() -> str:
    """
    优先用 npm 全局包版本识别 openclaw 版本。
    """
    if not command_exists("npm"):
        return ""
    rc, out = run_cmd(
        ["npm", "list", "-g", "openclaw", "--depth=0", "--no-update-notifier"],
        check=False,
        timeout=6,
    )
    if rc != 0 or not out:
        return ""

    for line in out.splitlines():
        if "openclaw@" in line:
            m = re.search(r"openclaw@([\w\.-]+)", line)
            if m:
                return m.group(1)
    return ""


def latest_openclaw_version(timeout_sec: int = 6) -> str:
    if not command_exists("npm"):
        return ""
    # 关闭 npm 更新提示，减少额外开销
    rc, out = run_cmd(
        ["npm", "view", "openclaw", "version", "--no-update-notifier"],
        check=False,
        timeout=timeout_sec,
    )
    return out.strip() if rc == 0 else ""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_url_text(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json, text/plain;q=0.9, */*;q=0.8")
    req.add_header("User-Agent", "opm-self-update/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_github_main_commit(timeout: int = 8) -> str:
    try:
        txt = fetch_url_text(SELF_COMMIT_API_URL, timeout=timeout)
        obj = json.loads(txt)
        return str(obj.get("sha") or "")
    except Exception:
        return ""


def check_self_update_status(timeout: int = 10) -> Dict[str, str]:
    local_hash = ""
    remote_hash = ""
    remote_commit = ""
    err = ""

    try:
        local_hash = file_sha256(THIS_SCRIPT)
    except Exception as e:
        err = f"本地哈希失败: {e}"

    remote_text = ""
    try:
        remote_text = fetch_url_text(SELF_RAW_URL, timeout=timeout)
        remote_hash = hashlib.sha256(remote_text.encode("utf-8")).hexdigest()
    except Exception as e:
        err = f"拉取远端脚本失败: {e}"

    remote_commit = fetch_github_main_commit(timeout=min(timeout, 8))

    has_update = False
    if local_hash and remote_hash:
        has_update = local_hash != remote_hash

    return {
        "repo": SELF_REPO_URL,
        "local": local_hash,
        "remote": remote_hash,
        "remote_commit": remote_commit,
        "has_update": "yes" if has_update else "no",
        "error": err,
        "fetched_text": remote_text,
    }


def update_self_script_from_github(timeout: int = 12) -> int:
    st = check_self_update_status(timeout=timeout)
    if st.get("error"):
        print(f"❌ 更新前检查失败：{st['error']}")
        return 1

    remote_text = st.get("fetched_text", "")
    if not remote_text.strip():
        print("❌ 远端脚本为空，已取消")
        return 1

    script_dir = os.path.dirname(THIS_SCRIPT)
    backup = os.path.join(script_dir, os.path.basename(THIS_SCRIPT) + ".bak")

    try:
        # 仅保留一个备份：始终覆盖同一路径（与主脚本同目录）
        shutil.copy2(THIS_SCRIPT, backup)
    except Exception as e:
        print(f"⚠️ 备份失败（继续尝试更新）：{e}")

    try:
        with open(THIS_SCRIPT, "w", encoding="utf-8") as f:
            f.write(remote_text)
        try:
            os.chmod(THIS_SCRIPT, 0o755)
        except Exception:
            pass
        print("✅ 脚本已更新")
        print(f"备份文件: {backup}")
        return 0
    except Exception as e:
        print(f"❌ 写入失败：{e}")
        return 1


def uninstall_self_script() -> int:
    script_path = THIS_SCRIPT
    script_dir = os.path.dirname(script_path)
    pycache_dir = os.path.join(script_dir, "__pycache__")
    backup_path = script_path + ".bak"

    print("\n⚠️ 卸载将删除以下内容：")
    print(f"- 脚本: {script_path}")
    print(f"- 备份: {backup_path}（若存在）")
    print(f"- 缓存目录: {pycache_dir}（若存在）")
    if not yes_no("确认继续卸载？", default=False):
        print("已取消卸载")
        return 0

    ok = True
    try:
        if os.path.exists(script_path):
            os.remove(script_path)
            print("✅ 已删除脚本")
        else:
            print("ℹ️ 脚本不存在，跳过")
    except Exception as e:
        ok = False
        print(f"❌ 删除脚本失败：{e}")

    try:
        if os.path.isfile(backup_path):
            os.remove(backup_path)
            print("✅ 已删除 .bak 备份")
        else:
            print("ℹ️ 未发现 .bak 备份，跳过")
    except Exception as e:
        ok = False
        print(f"❌ 删除 .bak 备份失败：{e}")

    try:
        if os.path.isdir(pycache_dir):
            shutil.rmtree(pycache_dir, ignore_errors=False)
            print("✅ 已删除 __pycache__")
        else:
            print("ℹ️ 未发现 __pycache__，跳过")
    except Exception as e:
        ok = False
        print(f"❌ 删除 __pycache__ 失败：{e}")

    return 0 if ok else 1



def get_cached_install_status() -> Dict[str, str]:
    return {
        "openclaw_ver": INSTALL_STATUS_CACHE.get("openclaw_ver", "") or "",
        "latest_ver": INSTALL_STATUS_CACHE.get("latest_ver", "") or "",
        "runtime": INSTALL_STATUS_CACHE.get("runtime", "") or "",
    }


def get_install_status(force: bool = False, probe_latest: bool = False) -> Dict[str, str]:
    """
    安装状态缓存：
    - 本地状态（openclaw版本、运行状态）默认 10 秒缓存
    - 远端最新版本（npm view）仅在需要时查询（probe_latest=True），并做 10 分钟缓存

    说明：进入“安装更新状态”页面不再默认触发 npm 查询，避免 CPU/负载突增。
    """
    now = time.time()

    # 本地状态
    if force or (now - float(INSTALL_STATUS_CACHE.get("ts", 0.0)) > 10.0):
        INSTALL_STATUS_CACHE["openclaw_ver"] = openclaw_version() or ""
        INSTALL_STATUS_CACHE["runtime"] = gateway_runtime_status_text()
        INSTALL_STATUS_CACHE["ts"] = now

    # 最新版本（网络）仅按需查询
    if probe_latest and (force or (now - float(INSTALL_STATUS_CACHE.get("latest_ts", 0.0)) > 600.0)):
        INSTALL_STATUS_CACHE["latest_ver"] = latest_openclaw_version(timeout_sec=6) or ""
        INSTALL_STATUS_CACHE["latest_ts"] = now

    return {
        "openclaw_ver": INSTALL_STATUS_CACHE.get("openclaw_ver", "") or "",
        "latest_ver": INSTALL_STATUS_CACHE.get("latest_ver", "") or "",
        "runtime": INSTALL_STATUS_CACHE.get("runtime", "未知") or "未知",
    }


def install_openclaw_official() -> int:
    """
    安装流程（精简版）：
    npm install -g openclaw@latest
    openclaw onboard --install-daemon
    gateway stop + start
    """
    if not command_exists("npm"):
        print("❌ 未检测到 npm，无法安装 OpenClaw")
        return 1

    print("\n➡️ 开始安装 OpenClaw...")
    rc1, out1 = run_cmd(["npm", "install", "-g", "openclaw@latest"], check=False)
    print((out1 or "")[-1200:])
    if rc1 != 0:
        return rc1

    rc2, out2 = run_cmd(["openclaw", "onboard", "--install-daemon"], check=False)
    print((out2 or "")[-1200:])

    # start_gateway 
    gateway_action("start")
    return 0 if rc2 == 0 else rc2


def check_openclaw_update_message(timeout_sec: int = 8) -> str:
    """检测: npm list + npm view 比较版本，只做检测提示。"""
    if not command_exists("npm"):
        return "❌ 未检测到 npm，无法检测更新"

    local_ver = openclaw_version()
    if not local_ver:
        return "❌ 未检测到本地 OpenClaw 版本"

    remote_ver = latest_openclaw_version(timeout_sec=timeout_sec)
    if not remote_ver:
        return "⚠️ 无法获取远端版本（网络或 npm 问题）"

    if local_ver != remote_ver:
        return f"⚠️ 检测到新版本：{remote_ver}（当前：{local_ver}）"
    return f"✅ 当前已是最新版本：{local_ver}"


def update_openclaw_official() -> int:
    # 保留函数名兼容其他调用，更新方式
    if not command_exists("npm"):
        print("❌ 未检测到 npm，无法更新 OpenClaw")
        return 1

    print("\n➡️ 开始更新 OpenClaw...")
    rc1, out1 = run_cmd(["npm", "install", "-g", "openclaw@latest"], check=False)
    print((out1 or "")[-1200:])
    if rc1 != 0:
        return rc1

    rc2, out2 = run_cmd(["openclaw", "onboard", "--install-daemon"], check=False)
    print((out2 or "")[-1200:])

    gateway_action("start")
    return 0 if rc2 == 0 else rc2


def uninstall_openclaw() -> int:
    """卸载核心命令：openclaw uninstall + npm uninstall -g openclaw"""
    print("\n⚠️ 卸载将移除 OpenClaw CLI 与服务，请谨慎。")
    if not yes_no("确认继续卸载？", default=False):
        print("已取消卸载")
        return 0

    rc_final = 0

    if command_exists("openclaw"):
        rc1, out1 = run_cmd(["openclaw", "uninstall"], check=False)
        print((out1 or "")[-1200:])
        if rc1 != 0:
            rc_final = rc1
    else:
        print("ℹ️ 未检测到 openclaw 命令，跳过 openclaw uninstall")

    if command_exists("npm"):
        rc2, out2 = run_cmd(["npm", "uninstall", "-g", "openclaw"], check=False)
        print((out2 or "")[-1200:])
        if rc2 != 0:
            rc_final = rc2
    else:
        print("ℹ️ 未检测到 npm，跳过 npm uninstall -g openclaw")

    return rc_final


def openclaw_doctor_fix() -> int:
    """检测修复：openclaw doctor --fix"""
    if not command_exists("openclaw"):
        print("❌ 未检测到 openclaw，无法执行健康检测")
        return 1
    rc, out = run_cmd(["openclaw", "doctor", "--fix"], check=False)
    print((out or "").strip())
    return rc


def cmd_oc_install(_args):
    return install_openclaw_official()


def cmd_oc_start(_args):
    return gateway_action("start")


def cmd_oc_stop(_args):
    return gateway_action("stop")


def cmd_oc_restart(_args):
    return gateway_action("restart")


def cmd_oc_status(_args):
    return gateway_action("status")


def cmd_oc_check_update(args):
    msg = check_openclaw_update_message(timeout_sec=getattr(args, "timeout", 8))
    print(msg)
    return 0 if msg.startswith("✅") or msg.startswith("⚠️") else 1


def cmd_oc_uninstall(_args):
    return uninstall_openclaw()


def cmd_oc_doctor(_args):
    return openclaw_doctor_fix()




def gateway_action(action: str) -> int:
    if action not in ("status", "start", "stop", "restart"):
        print("❌ 不支持的操作")
        return 1
    if not command_exists("openclaw"):
        print("❌ 未检测到 openclaw，无法操作 gateway")
        return 1

    # 主要 ：
    # - start: 先 stop 再 start，并 sleep 3 秒
    # - stop: 先尝试清理 tmux gateway 会话，再执行 stop
    # - restart: 等价于 stop + start
    # - status: 查看状态日志（openclaw status + gateway status + logs）
    if action == "start":
        _, out1 = run_cmd(["openclaw", "gateway", "stop"], check=False, timeout=12)
        rc2, out2 = run_cmd(["openclaw", "gateway", "start"], check=False, timeout=12)
        time.sleep(3)
        merged = "\n".join([x.strip() for x in (out1, out2) if x and x.strip()])
        print(merged.strip())
        return 0 if rc2 == 0 else rc2

    if action == "stop":
        run_cmd(["tmux", "kill-session", "-t", "gateway"], check=False, timeout=5)
        rc, out = run_cmd(["openclaw", "gateway", "stop"], check=False, timeout=12)
        print((out or "").strip())
        return rc

    if action == "restart":
        # 与“先停止再启动”保持一致：先清理 tmux gateway 会话，再 stop + start
        run_cmd(["tmux", "kill-session", "-t", "gateway"], check=False, timeout=5)
        _, out1 = run_cmd(["openclaw", "gateway", "stop"], check=False, timeout=12)
        rc2, out2 = run_cmd(["openclaw", "gateway", "start"], check=False, timeout=12)
        time.sleep(3)
        # 重启后额外做一次状态探测，确保配置已重新加载
        _, out3 = run_cmd(["openclaw", "gateway", "status"], check=False, timeout=12)
        merged = "\n".join([x.strip() for x in (out1, out2, out3) if x and x.strip()])
        print(merged.strip())
        return 0 if rc2 == 0 else rc2

    # status / logs
    rc1, out1 = run_cmd(["openclaw", "status"], check=False, timeout=12)
    rc2, out2 = run_cmd(["openclaw", "gateway", "status"], check=False, timeout=12)
    rc3, out3 = run_cmd(["openclaw", "logs"], check=False, timeout=20)
    show = "\n".join([x.strip() for x in (out1, out2, out3) if x and x.strip()])
    print(show.strip())
    return 0 if (rc1 == 0 and rc2 == 0 and rc3 == 0) else (rc3 if rc3 != 0 else (rc2 if rc2 != 0 else rc1))





def menu_install_update_status() -> int:
    # 进入此页面即检测一次状态
    primary = get_primary_model()
    st = get_install_status(force=True, probe_latest=True)

    while True:
        clear_screen()
        cur = st.get("openclaw_ver", "")
        latest = st.get("latest_ver", "")
        runtime = st.get("runtime", "")

        sep = MENU_SEP
        print(sep)
        print("openclaw设置")
        print(sep)
        print(f"默认模型: {primary}")
        print(f"OpenClaw版本: {cur if cur else '未安装'}")
        print(f"最新版本: {latest if latest else '未知（npm 不可用或网络问题）'}")
        print(f"运行状态: {runtime if runtime else '未知'}")
        print(sep)
        print("1. 安装 OpenClaw")
        print("2. 启动")
        print("3. 停止")
        print("4. 重启")
        print("5. 状态日志")
        print("6. 检测更新")
        print("7. 卸载")
        print("8. 检测/修复")
        print("q. 返回主菜单")
        print(sep)

        c = input("\n请输入编号: ").strip().lower()
        if c.lower() == "q":
            return 0

        if c == "1":
            if cur:
                print(f"✅ 已安装 OpenClaw，版本: {cur}")
            else:
                if yes_no("是否按 官网 风格安装 OpenClaw？", default=True):
                    rc = install_openclaw_official()
                    print("✅ 安装完成" if rc == 0 else "❌ 安装失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "2":
            rc = gateway_action("start")
            print("✅ 启动成功" if rc == 0 else "❌ 启动失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "3":
            rc = gateway_action("stop")
            print("✅ 停止成功" if rc == 0 else "❌ 停止失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "4":
            rc = gateway_action("restart")
            print("✅ 重启成功" if rc == 0 else "❌ 重启失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "5":
            rc = gateway_action("status")
            print("✅ 状态日志查看成功" if rc == 0 else "❌ 状态日志查看失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "6":
            msg = check_openclaw_update_message(timeout_sec=8)
            print(msg)
            # 保持原能力：如果有新版本可选择更新
            if msg.startswith("⚠️") and yes_no("是否立即更新到最新版本？", default=True):
                rc = update_openclaw_official()
                print("✅ 更新完成" if rc == 0 else "❌ 更新失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "7":
            rc = uninstall_openclaw()
            print("✅ 卸载完成" if rc == 0 else "❌ 卸载失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        if c == "8":
            rc = openclaw_doctor_fix()
            print("✅ 健康检测/修复完成" if rc == 0 else "❌ 健康检测/修复失败")
            st = get_install_status(force=True, probe_latest=True)
            primary = get_primary_model()
            pause_any_key()
            continue

        print("❌ 无效编号")
        pause_any_key()


def gateway_runtime_status_text() -> str:
    if not command_exists("openclaw"):
        return "未安装"

    # 轻量检测：先用进程名判断，避免每次调用 openclaw gateway status 的较高开销
    rc_pg, _ = run_cmd(["pgrep", "-f", "openclaw-gateway"], check=False, timeout=2)
    return "运行中" if rc_pg == 0 else "未运行"


def refresh_menu_status_once():
    try:
        cur = openclaw_version()
        latest = latest_openclaw_version()
        runtime = gateway_runtime_status_text()
        MENU_STATUS["openclaw_ver"] = cur or "未安装"
        MENU_STATUS["latest_ver"] = latest or "未知"
        MENU_STATUS["runtime"] = runtime
    except Exception:
        MENU_STATUS["openclaw_ver"] = MENU_STATUS.get("openclaw_ver") or "未知"
        MENU_STATUS["latest_ver"] = MENU_STATUS.get("latest_ver") or "未知"
        MENU_STATUS["runtime"] = MENU_STATUS.get("runtime") or "未知"


def start_menu_probe_if_needed():
    global _MENU_PROBE_STARTED
    if _MENU_PROBE_STARTED:
        return
    _MENU_PROBE_STARTED = True
    t = threading.Thread(target=refresh_menu_status_once, daemon=True)
    t.start()


def menu_refresh_status_blocking():
    refresh_menu_status_once()


def menu_status_signature() -> str:
    return "|".join(
        [
            str(MENU_STATUS.get("openclaw_ver", "")),
            str(MENU_STATUS.get("latest_ver", "")),
            str(MENU_STATUS.get("runtime", "")),
        ]
    )


def read_choice_with_auto_refresh(prompt: str, redraw_func):
    """
    主菜单输入：支持“状态变化后自动重绘一次”。
    仅在用户尚未开始输入时重绘，避免干扰其他交互。
    """
    if not sys.stdin.isatty():
        try:
            return input(prompt).strip()
        except EOFError:
            return "0"

    fd = None
    old = None
    try:
        import select
        import termios

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)

        # 关闭回显/规范模式，手动回显，避免重复数字与光标错位
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, termios.TCSADRAIN, new)

        buf = ""
        last_sig = menu_status_signature()
        refreshed_once = False

        sys.stdout.write(prompt)
        sys.stdout.flush()

        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.2)

            sig = menu_status_signature()
            if (not refreshed_once) and (sig != last_sig) and (buf == ""):
                refreshed_once = True
                last_sig = sig
                redraw_func()
                sys.stdout.write("\n" + prompt)
                sys.stdout.flush()

            if not r:
                continue

            ch = sys.stdin.read(1)
            if ch in ("\n", "\r"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return buf.strip()

            if ch in ("\x03",):  # Ctrl+C
                raise KeyboardInterrupt

            if ch in ("\x7f", "\b"):
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if ch and ch.isprintable():
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()

    except KeyboardInterrupt:
        raise
    except Exception:
        try:
            return input(prompt).strip()
        except EOFError:
            return "0"
    finally:
        try:
            if fd is not None and old is not None:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _wcs_width(s: str) -> int:
    w = 0
    for ch in str(s):
        # CJK/全角按2宽，其他按1宽
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _fit_line(text: str, inner: int) -> str:
    t = str(text)
    cur = _wcs_width(t)
    if cur > inner:
        out = ""
        used = 0
        for ch in t:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if used + cw > max(0, inner - 1):
                break
            out += ch
            used += cw
        t = out + "…"
        cur = _wcs_width(t)
    return t + " " * max(0, inner - cur)


def _fit_line_center(text: str, inner: int) -> str:
    t = str(text)
    cur = _wcs_width(t)
    if cur > inner:
        out = ""
        used = 0
        for ch in t:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if used + cw > max(0, inner - 1):
                break
            out += ch
            used += cw
        t = out + "…"
        cur = _wcs_width(t)
    pad_total = max(0, inner - cur)
    left = pad_total // 2
    right = pad_total - left
    return " " * left + t + " " * right


def _box_print(lines: List[str], width: int = 62):
    inner = width - 4
    print("╔" + "═" * (width - 2) + "╗")
    for i, raw in enumerate(lines):
        if raw == "__SEP__":
            print("╠" + "═" * (width - 2) + "╣")
        elif isinstance(raw, str) and raw.startswith("__CENTER__:"):
            text = raw.split(":", 1)[1]
            print("║ " + _fit_line_center(text, inner) + " ║")
        else:
            print("║ " + _fit_line(raw, inner) + " ║")
    print("╚" + "═" * (width - 2) + "╝")


def menu_self_update_uninstall() -> int:
    st = check_self_update_status(timeout=10)

    while True:
        clear_screen()
        print(MENU_SEP)
        print("opm脚本更新/卸载")
        print(MENU_SEP)
        print(f"仓库地址: {SELF_REPO_URL}")
        if st.get("error"):
            print(f"检测结果: ❌ {st['error']}")
        else:
            upd = "是" if st.get("has_update") == "yes" else "否"
            print(f"是否有更新: {upd}")
        print(MENU_SEP)
        print("1. 重新检测更新")
        print("2. 更新脚本")
        print("3. 卸载脚本")
        print("q. 返回主菜单")
        print(MENU_SEP)

        c = input("\n请输入编号: ").strip()
        if c.lower() == "q":
            return 0

        if c == "1":
            st = check_self_update_status(timeout=10)
            pause_any_key()
            continue

        if c == "2":
            if not yes_no("确认从 GitHub 更新当前脚本？", default=True):
                print("已取消更新")
                pause_any_key()
                continue
            rc = update_self_script_from_github(timeout=12)
            if rc == 0:
                print("✅ 更新完成（建议重新运行脚本）")
            else:
                print("❌ 更新失败")
            st = check_self_update_status(timeout=10)
            pause_any_key()
            continue

        if c == "3":
            rc = uninstall_self_script()
            if rc == 0:
                print("✅ 卸载完成，程序即将退出")
                time.sleep(1)
                return 99
            print("❌ 卸载失败")
            pause_any_key()
            continue

        print("❌ 无效编号")
        pause_any_key()


def menu_loop() -> int:
    def draw_main():
        clear_screen()
        sep = MENU_SEP
        print(sep)
        print("OpenClaw Manager")
        print(sep)
        print("1. openclaw设置")
        print("2. 列出 provider（list）")
        print("3. 检测 API 可用性（check）")
        print("4. 添加 provider（add）")
        print("5. 同步模型注册（sync）")
        print("6. 切换默认模型（switch）")
        print("7. 查看 provider 模型（models）")
        print("8. 删除 provider（remove）")
        print("9. 安装/修复别名（alias-install）")
        print("q. 退出")
        print("00. 更新/卸载（opm脚本）")
        print(sep)
        print("提示：命令行帮助用 opm --cli-help；返回用 q")

    while True:
        draw_main()
        choice = input("\n请输入编号: ").strip()
        print()

        if choice.lower() == "q":
            print("👋 已退出 OPM")
            return 0

        if choice == "00":
            rc = menu_self_update_uninstall()
            if rc == 99:
                return 0
            continue

        if choice == "1":
            menu_install_update_status()
            continue

        if choice == "2":
            run_action(cmd_list, argparse.Namespace())
            pause_any_key()
            continue

        if choice == "3":
            print("当前 provider 列表：")
            run_action(cmd_list, argparse.Namespace())
            providers_sel = prompt_text("检测本地已添加API（0=全部；支持 1 2 或 名称；输入s跳过到新API检测；输入q返回）: ", "0", allow_zero_cancel=True)
            if providers_sel is None:
                continue
            timeout = prompt_text("超时秒数(默认10，输入q返回): ", "10", allow_zero_cancel=True)
            if timeout is None:
                continue
            ua = prompt_text(f"User-Agent(默认 {DEFAULT_UA}，输入q返回): ", DEFAULT_UA, allow_zero_cancel=True)
            if ua is None:
                continue

            skipped_local = str(providers_sel).strip().lower() == "s"

            if not skipped_local:
                print("\n➡️ 开始检测本地已添加 API...")
                run_action(
                    cmd_check,
                    argparse.Namespace(
                        base_url="",
                        api_key="",
                        providers=providers_sel,
                        new_api=False,
                        timeout=to_int(timeout, 10),
                        ua=ua,
                    ),
                )
            else:
                print("\n⏭️ 已跳过本地 API 检测")

            if yes_no("是否继续检测新添加 API？", default=True):
                print("\n➡️ 开始检测新 API...")
                run_action(
                    cmd_check,
                    argparse.Namespace(
                        base_url="",
                        api_key="",
                        providers="0",
                        new_api=True,
                        timeout=to_int(timeout, 10),
                        ua=ua,
                    ),
                )

            pause_any_key()
            continue


        if choice == "4":
            validate = yes_no("是否开启 chat 可用性验证(--validate-chat)", default=True)
            prune_hint = "（添加场景无 prune）"
            print(f"validate-chat: {'开' if validate else '关'} {prune_hint}")
            timeout = prompt_text("超时秒数(默认10，输入q返回): ", "10", allow_zero_cancel=True)
            if timeout is None:
                continue
            max_probe = prompt_text("max-probe(0=全量，输入0继续；若要返回请输入 q): ", "0", allow_zero_cancel=False)
            if str(max_probe).lower() == "q":
                continue
            ua = prompt_text(f"User-Agent(默认 {DEFAULT_UA}，输入q返回): ", DEFAULT_UA, allow_zero_cancel=True)
            if ua is None:
                continue
            no_restart = yes_no("仅改配置不重启 gateway(--no-restart)", default=False)
            run_action(
                cmd_add,
                argparse.Namespace(
                    name="",
                    base_url="",
                    api_key="",
                    api="openai-completions",
                    inputs="text image",
                    context_window=131072,
                    max_tokens=8192,
                    timeout=to_int(timeout, 10),
                    ua=ua,
                    validate_chat=validate,
                    max_probe=to_int(max_probe, 0),
                    no_restart=no_restart,
                ),
            )
            pause_any_key()
            continue

        if choice == "5":
            print("当前 provider 列表：")
            run_action(cmd_list, argparse.Namespace())
            providers = prompt_text("要同步的 provider（序号/名称，空=全部，输入q返回）: ", "", allow_zero_cancel=True)
            if providers is None:
                continue
            print("同步说明：默认仅同步可用模型，并自动移除不可用/失效模型")
            timeout = prompt_text("超时秒数(默认10，输入q返回): ", "10", allow_zero_cancel=True)
            if timeout is None:
                continue
            max_probe = prompt_text("max-probe(0=全量): ", "0", allow_zero_cancel=False)
            ua = prompt_text(f"User-Agent(默认 {DEFAULT_UA}，输入q返回): ", DEFAULT_UA, allow_zero_cancel=True)
            if ua is None:
                continue
            no_restart = yes_no("仅改配置不重启 gateway(--no-restart)", default=False)
            run_action(
                cmd_sync,
                argparse.Namespace(
                    providers=providers,
                    validate_chat=True,
                    prune=True,
                    timeout=to_int(timeout, 10),
                    ua=ua,
                    max_probe=to_int(max_probe, 0),
                    no_restart=no_restart,
                ),
            )
            pause_any_key()
            continue

        if choice == "6":
            session = yes_no("是否同步切换当前会话模型(--session)", default=True)
            no_restart = yes_no("仅改配置不重启 gateway(--no-restart)", default=False)
            run_action(
                cmd_switch,
                argparse.Namespace(provider="", model="", session=session, no_restart=no_restart),
            )
            pause_any_key()
            continue

        if choice == "7":
            print("当前 provider 列表：")
            run_action(cmd_list, argparse.Namespace())
            provider = prompt_text("请输入 provider 序号或名称（支持多个，空格分隔，输入q返回）: ", "", allow_zero_cancel=True)
            if provider is None:
                continue
            check_avail = yes_no("是否检查模型可用性（默认是）", default=True)
            timeout = prompt_text("检测超时秒数(默认10，输入q返回): ", "10", allow_zero_cancel=True)
            if timeout is None:
                continue
            max_probe = prompt_text("每个provider最多检测模型数(0=全部，默认0): ", "0", allow_zero_cancel=False)
            ua = prompt_text(f"User-Agent(默认 {DEFAULT_UA}，输入q返回): ", DEFAULT_UA, allow_zero_cancel=True)
            if ua is None:
                continue
            run_action(
                cmd_models,
                argparse.Namespace(
                    provider=provider,
                    no_check=(not check_avail),
                    timeout=to_int(timeout, 10),
                    ua=ua,
                    max_probe=to_int(max_probe, 0),
                ),
            )
            pause_any_key()
            continue

        if choice == "8":
            print("当前 provider 列表：")
            run_action(cmd_list, argparse.Namespace())
            names = prompt_text("要删除的 provider（序号/名称，空格分隔，输入q返回）: ", "", allow_zero_cancel=True)
            if names is None:
                continue
            no_restart = yes_no("仅改配置不重启 gateway(--no-restart)", default=False)
            run_action(cmd_remove, argparse.Namespace(name="", names=names, no_restart=no_restart))
            pause_any_key()
            continue

        if choice == "9":
            name = prompt_text("别名名称(默认 opm，输入q返回): ", "opm", allow_zero_cancel=True)
            if name is None:
                continue
            path = prompt_text("安装路径(默认 /usr/local/bin 或 ~/bin，输入q返回): ", "", allow_zero_cancel=True)
            if path is None:
                continue
            run_action(cmd_alias, argparse.Namespace(name=name, path=path))
            pause_any_key()
            continue

        print("❌ 无效编号，请重试")
        pause_any_key()


def build_parser():
    p = argparse.ArgumentParser(
        description="OpenClaw provider 管理脚本（CLI 模式）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm check --base-url https://api.xxxx.com/v1 --api-key <key>\n"
            "  opm ad --name demo1 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat\n"
            "  opm sy --providers demo2 --validate-chat --prune\n"
            "  opm sw --session\n"
            "  opm oc-install\n"
            "  opm oc-start\n"
            "  opm oc-check-update\n"
            "\n"
            "提示:\n"
            "  opm              进入数字菜单主界面\n"
            "  opm --cli-help   查看 CLI 总帮助（等价于 --help）\n"
            "  opm <子命令> -h  查看该命令用法与示例\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser(
        "check",
        aliases=["ck"],
        help="检测 API 可用性（默认检测本地 provider，可检测新 API）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm check                         # 默认检测本地全部 provider(API)\n"
            "  opm check --providers \"1 2\"     # 仅检测指定 provider\n"
            "  opm check --providers demo1 demo2\n"
            "  opm check --new-api --base-url https://api.xxx.com/v1 --api-key sk-xxxx\n"
            "  opm ck --timeout 12 --ua curl/8.5.0\n"
        ),
    )
    p_check.add_argument("--base-url", default="", help="新 API 模式可用：例如 https://api.xxx.com/v1")
    p_check.add_argument("--api-key", default="", help="新 API 模式可用：API Key")
    p_check.add_argument("--providers", nargs="+", default=["0"], help="本地 provider 选择：0=全部；支持序号/名称，可多项")
    p_check.add_argument("--new-api", action="store_true", help="检测新 API（base-url/api-key）")
    p_check.add_argument("--timeout", type=int, default=10, help="请求超时秒数")
    p_check.add_argument("--ua", default=DEFAULT_UA, help=f"User-Agent，默认 {DEFAULT_UA}")
    p_check.set_defaults(func=cmd_check)

    p_add = sub.add_parser(
        "add",
        aliases=["ad"],
        help="添加 provider 并注册其全部模型（支持交互输入）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm ad --name myp --base-url https://api.xxx.com/v1 --api-key <key>\n"
            "  opm ad --name demo2 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat\n"
            "  opm ad --name demo2 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat --max-probe 20\n"
        ),
    )
    p_add.add_argument("--name", default="", help="provider 名称，例如: myprovider；不填则交互输入")
    p_add.add_argument("--base-url", default="", help="例如: https://api.xxx.com/v1；不填则交互输入")
    p_add.add_argument("--api-key", default="", help="API Key；不填则交互输入")
    p_add.add_argument("--api", default="openai-completions", help="API 类型，默认 openai-completions")
    p_add.add_argument("--inputs", default="text image", help="模型输入类型，空格/逗号分隔，默认 text image")
    p_add.add_argument("--context-window", type=int, default=131072, help="contextWindow 默认 131072")
    p_add.add_argument("--max-tokens", type=int, default=8192, help="maxTokens 默认 8192")
    p_add.add_argument("--timeout", type=int, default=10, help="请求超时秒数")
    p_add.add_argument("--ua", default=DEFAULT_UA, help=f"User-Agent，默认 {DEFAULT_UA}")
    p_add.add_argument("--validate-chat", action="store_true", help="添加前验证 chat/completions，仅注册可调用模型")
    p_add.add_argument("--max-probe", type=int, default=0, help="验证模型数量上限，0=验证全部")
    p_add.add_argument("--no-restart", action="store_true", help="仅改配置，不重启 gateway")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser(
        "remove",
        aliases=["rm"],
        help="删除 provider 并清理已注册模型（支持批量/序号）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm remove --name demo1\n"
            "  opm rm --names 1 3 demo2\n"
        ),
    )
    p_rm.add_argument("--name", help="单个 provider 名称或序号（可留空，优先 names）", default="")
    p_rm.add_argument("--names", help="批量删除，支持序号/名称，空格分隔，例如 1 3 myprovider", default="")
    p_rm.add_argument("--no-restart", action="store_true", help="仅改配置，不重启 gateway")
    p_rm.set_defaults(func=cmd_remove)

    p_sync = sub.add_parser(
        "sync",
        aliases=["sy"],
        help="把 provider 的模型注册进 agents.defaults.models，可选单个/多选/序号",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm sy --providers demo2 --validate-chat\n"
            "  opm sy --providers demo2 --validate-chat --prune\n"
            "  opm sy --providers 2 3 --validate-chat --timeout 12 --max-probe 50\n"
        ),
    )
    p_sync.add_argument("--providers", default="", help="指定要同步的 provider（序号或名称，空格分隔）；空则全部")
    p_sync.add_argument("--validate-chat", action="store_true", help="同步前验证 chat/completions，仅注册可调用模型")
    p_sync.add_argument("--prune", action="store_true", help="配合 --validate-chat：清理已注册但验证失败的模型")
    p_sync.add_argument("--timeout", type=int, default=10, help="验证请求超时秒数")
    p_sync.add_argument("--ua", default=DEFAULT_UA, help=f"验证请求 User-Agent，默认 {DEFAULT_UA}")
    p_sync.add_argument("--max-probe", type=int, default=0, help="每个 provider 验证模型数量上限，0=验证全部")
    p_sync.add_argument("--no-restart", action="store_true", help="仅改配置，不重启 gateway")
    p_sync.set_defaults(func=cmd_sync)

    p_list = sub.add_parser(
        "list",
        aliases=["ls"],
        help="列出 provider 与注册状态（带序号）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm list\n"
            "  opm ls\n"
        ),
    )
    p_list.set_defaults(func=cmd_list)

    p_models = sub.add_parser(
        "models",
        aliases=["md"],
        help="列出 provider 模型（支持多个 provider，可检测可用性）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm models --provider demo1\n"
            "  opm md --provider \"1 2\"\n"
            "  opm md --provider \"demo1 demo2\" --no-check\n"
            "  opm md --provider \"1 2\" --timeout 12 --max-probe 20\n"
        ),
    )
    p_models.add_argument("--provider", required=True, help="provider 序号或名称，支持多个，空格分隔")
    p_models.add_argument("--no-check", action="store_true", help="不检测模型可用性（默认检测）")
    p_models.add_argument("--timeout", type=int, default=10, help="可用性检测超时秒数，默认10")
    p_models.add_argument("--ua", default=DEFAULT_UA, help=f"检测 User-Agent，默认 {DEFAULT_UA}")
    p_models.add_argument("--max-probe", type=int, default=0, help="每个 provider 最多检测多少模型，0=全部")
    p_models.set_defaults(func=cmd_models)

    p_switch = sub.add_parser(
        "switch",
        aliases=["sw"],
        help="切换默认模型：provider + 模型序号/名称",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm switch --provider demo1 --model gpt-4o-mini\n"
            "  opm sw --provider 1 --model 2 --session\n"
        ),
    )
    p_switch.add_argument("--provider", default="", help="provider 序号或名称；不填则交互选择")
    p_switch.add_argument("--model", default="", help="模型序号或模型 id，空则取该 provider 第一个")
    p_switch.add_argument("--session", action="store_true", help="同步切换当前会话模型（/model <provider/model>）")
    p_switch.add_argument("--no-restart", action="store_true", help="仅改配置，不重启 gateway")
    p_switch.set_defaults(func=cmd_switch)

    p_alias = sub.add_parser(
        "alias-install",
        aliases=["ai"],
        help="安装命令别名（软链接），默认 opm",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm alias-install --name opm\n"
            "  opm ai --name opm --path /usr/local/bin\n"
        ),
    )
    p_alias.add_argument("--name", default="opm", help="别名名称，默认 opm")
    p_alias.add_argument("--path", default="", help="可选：指定安装路径（默认为 /usr/local/bin，失败则 ~/bin）")
    p_alias.set_defaults(func=cmd_alias)



    # openclaw设置（非交互命令）
    p_oc_install = sub.add_parser(
        "oc-install",
        help="安装 OpenClaw",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-install\n"
        ),
    )
    p_oc_install.set_defaults(func=cmd_oc_install)

    p_oc_start = sub.add_parser(
        "oc-start",
        help="启动 gateway",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-start\n"
        ),
    )
    p_oc_start.set_defaults(func=cmd_oc_start)

    p_oc_stop = sub.add_parser(
        "oc-stop",
        help="停止 gateway",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-stop\n"
        ),
    )
    p_oc_stop.set_defaults(func=cmd_oc_stop)

    p_oc_restart = sub.add_parser(
        "oc-restart",
        help="重启 gateway",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-restart\n"
        ),
    )
    p_oc_restart.set_defaults(func=cmd_oc_restart)

    p_oc_status = sub.add_parser(
        "oc-status",
        help="日志查看",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-status\n"
        ),
    )
    p_oc_status.set_defaults(func=cmd_oc_status)

    p_oc_check = sub.add_parser(
        "oc-check-update",
        help="检测 OpenClaw 更新",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-check-update\n"
            "  opm oc-check-update --timeout 12\n"
        ),
    )
    p_oc_check.add_argument("--timeout", type=int, default=8, help="检测超时秒数，默认8")
    p_oc_check.set_defaults(func=cmd_oc_check_update)

    p_oc_uninstall = sub.add_parser(
        "oc-uninstall",
        help="卸载 OpenClaw",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-uninstall\n"
        ),
    )
    p_oc_uninstall.set_defaults(func=cmd_oc_uninstall)

    p_oc_doctor = sub.add_parser(
        "oc-doctor",
        help="检测/修复",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  opm oc-doctor\n"
        ),
    )
    p_oc_doctor.set_defaults(func=cmd_oc_doctor)

    return p


def main():
    argv = sys.argv[1:]

    # 无参数：进入数字菜单
    if not argv:
        return menu_loop()

    # 非交互 CLI 帮助（总帮助）
    if "--cli-help" in argv:
        argv = [x for x in argv if x != "--cli-help"] + ["--help"]

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 已退出 OPM")
        return 130
    except SystemExit as e:
        # argparse 的正常退出（例如 --help / 子命令 -h）
        return int(e.code) if isinstance(e.code, int) else 0
    except Exception as e:
        eprint(f"❌ 执行失败: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 已退出 OPM")
        sys.exit(130)

