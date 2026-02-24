#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw Provider Manager

功能：
1) 检测 API 是否可用（check）
2) 添加 provider 并自动注册该 provider 的全部模型（add）
3) 删除 provider（支持批量/序号/空格分隔）并清理已注册模型（remove）
4) 同步 provider 的模型到 agents.defaults.models（sync，可选单个/多选，空格分隔）
5) 列出当前 provider（list，带序号），以及单个 provider 模型列表（models）
6) 快速切换默认模型（switch：按 provider+模型序号/名称）
7) 安装/自定义别名（alias-install，默认别名 pm）

示例：
  python3 provider_manager.py check --base-url https://api.xxx.com/v1 --api-key sk-xxx
  python3 provider_manager.py add --name myp --base-url https://api.xxx.com/v1 --api-key sk-xxx
  python3 provider_manager.py remove --names 1 3 myp
  python3 provider_manager.py sync --providers 1 myp
  python3 provider_manager.py models --provider 3
  python3 provider_manager.py switch --provider 3 --model 2
  python3 provider_manager.py alias-install --name pm
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
import unicodedata
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

CONFIG_PATH = "/root/.openclaw/openclaw.json"
DEFAULT_UA = "curl/8.5.0"
THIS_SCRIPT = os.path.abspath(__file__)
MENU_SEP = "=" * 33
MENU_STATUS = {
    "openclaw_ver": "检测中...",
    "latest_ver": "检测中...",
    "runtime": "检测中...",
}
_MENU_PROBE_STARTED = False


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_cmd(cmd, check=True):
    """Run command without shell to avoid quoting issues."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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


def filter_models_by_chat(base_url: str, api_key: str, model_ids: List[str], timeout: int, ua: str, max_probe: int = 0) -> Tuple[List[str], List[str]]:
    """
    Validate model ids by probing /chat/completions.
    max_probe=0 means probe all models.
    Returns (ok_models, bad_models).
    """
    ids = model_ids[:]
    if max_probe and max_probe > 0:
        ids = ids[:max_probe]
    ok, bad = [], []
    for mid in ids:
        succ, code, _ = probe_chat_model(base_url, api_key, mid, timeout=timeout, ua=ua)
        if succ:
            ok.append(mid)
        else:
            bad.append(mid)
    # if limited probe, keep unprobed models as-is (append after validated ok)
    if max_probe and max_probe > 0 and len(model_ids) > max_probe:
        remaining = model_ids[max_probe:]
        ok.extend(remaining)
    return ok, bad


def config_set(path_key: str, obj):
    payload = json.dumps(obj, ensure_ascii=False)
    run_cmd(["openclaw", "config", "set", path_key, "--json", payload], check=True)


def restart_gateway():
    rc, _ = run_cmd(["openclaw", "gateway", "restart"], check=False)
    if rc == 0:
        print("✅ gateway 已重启（openclaw gateway restart）")
        return True

    eprint("⚠️ openclaw gateway restart 失败，尝试兜底重启...")
    try:
        rc, out = run_cmd(["pgrep", "-f", "openclaw-gateway"], check=False)
        if rc == 0 and out.strip():
            for pid_s in out.strip().splitlines():
                try:
                    os.kill(int(pid_s.strip()), signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(2)
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
    base_url = (args.base_url or "").strip()
    api_key = (args.api_key or "").strip()
    ua = args.ua.strip() or DEFAULT_UA

    # 交互式输入：支持直接执行 `pm check`
    if not base_url:
        base_url = input("请输入 base-url (例如 https://api.xxx.com/v1): ").strip()
    if not api_key:
        api_key = getpass.getpass("请输入 api-key: ").strip()

    ok, status, model_ids, detail = fetch_models(base_url, api_key, args.timeout, ua)
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


def cmd_add(args):
    provider = (args.name or "").strip()
    if not provider:
        provider = input("请输入 provider 名称 (例如 myprovider): ").strip()
    if not provider:
        raise ValueError("provider name 不能为空")

    base_url = (args.base_url or "").strip()
    if not base_url:
        base_url = input("请输入 base-url (例如 https://api.xxx.com/v1): ").strip()
    base_url = normalize_base_url(base_url)

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = getpass.getpass("请输入 api-key: ").strip()

    ua = args.ua.strip() or DEFAULT_UA

    ok, status, model_ids, detail = fetch_models(base_url, api_key, args.timeout, ua)
    if not ok:
        print("❌ 先检测 API 失败，已取消添加")
        print(f"HTTP: {status}")
        print(f"详情: {detail[:800]}")
        return 2

    # 可选：仅保留 chat/completions 实测可用模型，避免“能列出但调用404”
    bad_models = []
    if args.validate_chat:
        model_ids, bad_models = filter_models_by_chat(
            base_url, api_key, model_ids, timeout=args.timeout, ua=ua, max_probe=args.max_probe
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
    added = 0
    removed = 0
    for pname in targets:
        pobj = cfg.get("models", {}).get("providers", {}).get(pname) or {}

        # 可选：先用 chat/completions 验证 provider 模型可用性
        if args.validate_chat:
            mids = [m.get("id") for m in (pobj.get("models", []) or []) if isinstance(m, dict) and m.get("id")]
            ok_ids, bad_ids = filter_models_by_chat(
                pobj.get("baseUrl", ""),
                pobj.get("apiKey", ""),
                mids,
                timeout=args.timeout,
                ua=args.ua,
                max_probe=args.max_probe,
            )
            valid_ids = ok_ids
            if bad_ids:
                print(f"⚠️ {pname} 过滤不可用模型: {len(bad_ids)}")
                if args.prune:
                    bad_keys = {f"{pname}/{x}" for x in bad_ids}
                    for k in list(defaults_models.keys()):
                        if k in bad_keys:
                            defaults_models.pop(k, None)
                            removed += 1
        else:
            valid_ids = [m.get("id") for m in (pobj.get("models", []) or []) if isinstance(m, dict) and m.get("id")]

        for mid in valid_ids:
            key = f"{pname}/{mid}"
            if key not in defaults_models:
                defaults_models[key] = {}
                added += 1

    config_set("agents.defaults.models", defaults_models)

    if not args.no_restart:
        restart_gateway()

    if removed:
        print(f"✅ 同步完成，新增注册模型: {added}，清理失效模型: {removed}")
    else:
        print(f"✅ 同步完成，新增注册模型: {added}")
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
    selected = parse_selection(args.provider, names_sorted)
    pname = selected[0] if selected else (args.provider.strip() if args.provider else "")
    if not pname:
        print("❌ 需要 --provider 指定（序号或名称）")
        return 1
    providers = cfg.get("models", {}).get("providers", {})
    pobj = providers.get(pname)
    if not pobj:
        print(f"❌ provider 不存在: {pname}")
        return 1
    models = pobj.get("models", []) or []
    print(f"provider: {pname}, 模型数: {len(models)}")
    for i, m in enumerate(models, 1):
        mid = m.get("id") if isinstance(m, dict) else str(m)
        print(f"{i}. {mid}")
    return 0


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
            inp = input("provider: ").strip()
        except EOFError:
            return 1
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
            inp = input("模型序号或名称（空回车默认第1个）: ").strip()
        except EOFError:
            return 1
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


def install_alias(alias_name: str = "pm", path_override: str = ""):
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
    name = args.name.strip() or "pm"
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
    s = input(f"{prompt} [{tip}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true")


def to_int(raw: str, default: int) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def run_action(func, args_obj) -> int:
    try:
        return int(func(args_obj) or 0)
    except Exception as e:
        eprint(f"❌ 执行失败: {e}")
        return 1


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def openclaw_version() -> str:
    if not command_exists("openclaw"):
        return ""
    rc, out = run_cmd(["openclaw", "--version"], check=False)
    return out.strip() if rc == 0 else ""


def latest_openclaw_version() -> str:
    if not command_exists("npm"):
        return ""
    rc, out = run_cmd(["npm", "view", "openclaw", "version"], check=False)
    return out.strip() if rc == 0 else ""


def install_openclaw_official() -> int:
    print("\n➡️ 使用官方安装脚本安装 OpenClaw...")
    rc, out = run_cmd(
        [
            "bash",
            "-lc",
            "curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install.sh | bash -s -- --no-onboard",
        ],
        check=False,
    )
    print(out[-1200:] if out else "")
    return rc


def update_openclaw_official() -> int:
    print("\n➡️ 使用官方安装脚本更新 OpenClaw...")
    rc, out = run_cmd(
        ["bash", "-lc", "curl -fsSL https://openclaw.ai/install.sh | bash"],
        check=False,
    )
    print(out[-1200:] if out else "")
    return rc


def uninstall_openclaw() -> int:
    if not command_exists("openclaw"):
        print("ℹ️ 未检测到 openclaw，无需卸载")
        return 0

    print("\n⚠️ 卸载将移除 OpenClaw CLI 与服务，请谨慎。")
    if not yes_no("确认继续卸载？", default=False):
        print("已取消卸载")
        return 0

    # 优先非交互卸载
    rc, out = run_cmd(["openclaw", "uninstall", "--all", "--yes", "--non-interactive"], check=False)
    if rc != 0:
        print("⚠️ 非交互卸载失败，尝试普通卸载...")
        rc, out = run_cmd(["openclaw", "uninstall"], check=False)
    print(out[-1200:] if out else "")
    return rc


def gateway_action(action: str) -> int:
    if action not in ("status", "start", "stop", "restart"):
        print("❌ 不支持的操作")
        return 1
    if not command_exists("openclaw"):
        print("❌ 未检测到 openclaw，无法操作 gateway")
        return 1
    rc, out = run_cmd(["openclaw", "gateway", action], check=False)
    print(out.strip())
    return rc


def menu_install_update_status() -> int:
    while True:
        clear_screen()
        cur = openclaw_version()
        latest = latest_openclaw_version()

        sep = MENU_SEP
        print(sep)
        print("安装更新状态")
        print(sep)
        print(f"当前安装: {'是' if cur else '否'}" + (f"（版本: {cur}）" if cur else ""))
        print(f"最新版本: {latest if latest else '未知（npm 不可用或网络失败）'}")
        print(sep)
        print("1. 安装/卸载 OpenClaw")
        print("2. 检测更新并可更新")
        print("3. Gateway 启动/停止/重启/状态")
        print("0. 返回主菜单")
        print(sep)

        c = input("\n请输入编号: ").strip()
        if c == "0":
            return 0

        if c == "1":
            if cur:
                print(f"✅ 已安装 OpenClaw，版本: {cur}")
                if yes_no("是否卸载 OpenClaw？", default=False):
                    rc = uninstall_openclaw()
                    print("✅ 卸载完成" if rc == 0 else "❌ 卸载失败")
            else:
                print("ℹ️ 未安装 OpenClaw")
                if yes_no("是否使用官方脚本安装？", default=True):
                    rc = install_openclaw_official()
                    print("✅ 安装完成" if rc == 0 else "❌ 安装失败")
            pause_any_key()
            continue

        if c == "2":
            if not cur:
                print("❌ 未安装 OpenClaw，无法更新")
                pause_any_key()
                continue
            if not latest:
                print("⚠️ 无法获取最新版本（npm 或网络问题）")
                pause_any_key()
                continue
            print(f"当前版本: {cur}")
            print(f"最新版本: {latest}")
            if cur == latest:
                print("✅ 已是最新版本")
            else:
                print("⚠️ 检测到新版本")
                if yes_no("是否立即更新？", default=True):
                    rc = update_openclaw_official()
                    print("✅ 更新完成" if rc == 0 else "❌ 更新失败")
            pause_any_key()
            continue

        if c == "3":
            print("\n1. status  2. start  3. stop  4. restart")
            x = input("请选择操作: ").strip()
            amap = {"1": "status", "2": "start", "3": "stop", "4": "restart"}
            act = amap.get(x, "")
            if not act:
                print("❌ 无效操作")
            else:
                rc = gateway_action(act)
                if rc == 0:
                    print("✅ 执行成功")
                else:
                    print("❌ 执行失败")
            pause_any_key()
            continue

        print("❌ 无效编号")
        pause_any_key()


def gateway_runtime_status_text() -> str:
    if not command_exists("openclaw"):
        return "未安装"

    rc, out = run_cmd(["openclaw", "gateway", "status"], check=False)
    txt = (out or "").lower()

    if rc == 0:
        if ("running" in txt) or ("active" in txt) or ("已运行" in out):
            return "运行中"
        if ("not running" in txt) or ("stopped" in txt) or ("inactive" in txt) or ("未运行" in out):
            return "未运行"
        return "正常"

    if ("not running" in txt) or ("stopped" in txt) or ("inactive" in txt) or ("未运行" in out):
        return "未运行"
    return "异常"


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
    # 非 TTY 环境回退普通输入
    if not sys.stdin.isatty():
        try:
            return input(prompt).strip()
        except EOFError:
            return "0"

    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        buf = ""
        last_sig = menu_status_signature()

        sys.stdout.write(prompt)
        sys.stdout.flush()

        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.2)

            # 状态变化时自动重绘（仅在还未输入字符时，避免影响输入体验）
            sig = menu_status_signature()
            if sig != last_sig and not buf:
                last_sig = sig
                redraw_func()
                sys.stdout.write("\n" + prompt)
                sys.stdout.flush()

            if r:
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return buf.strip()
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
    except Exception:
        try:
            return input(prompt).strip()
        except EOFError:
            return "0"
    finally:
        try:
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


def menu_loop() -> int:
    start_menu_probe_if_needed()

    def draw_main():
        clear_screen()
        primary = get_primary_model()
        sep = MENU_SEP
        print(sep)
        print("OpenClaw Provider Manager")
        print(sep)
        print(f"默认模型: {primary}")
        print(f"OpenClaw版本: {MENU_STATUS.get('openclaw_ver', '检测中...')}")
        print(f"最新版本: {MENU_STATUS.get('latest_ver', '检测中...')}")
        print(f"运行状态: {MENU_STATUS.get('runtime', '检测中...')}")
        print(sep)
        print("1. 安装更新状态")
        print("2. 列出 provider（list）")
        print("3. 检测 API 可用性（check）")
        print("4. 添加 provider（add）")
        print("5. 同步模型注册（sync）")
        print("6. 切换默认模型（switch）")
        print("7. 查看 provider 模型（models）")
        print("8. 删除 provider（remove）")
        print("9. 安装/修复别名（alias-install）")
        print("0. 退出")
        print(sep)
        print("提示：命令行帮助请用 pm --cli-help")

    while True:
        draw_main()
        choice = read_choice_with_auto_refresh("\n请输入编号: ", draw_main)
        print()
        if choice == "1":
            menu_install_update_status()
            continue

        if choice == "2":
            run_action(cmd_list, argparse.Namespace())
            pause_any_key()
            continue

        if choice == "3":
            timeout = input("超时秒数(默认10): ").strip() or "10"
            ua = input(f"User-Agent(默认 {DEFAULT_UA}): ").strip() or DEFAULT_UA
            run_action(
                cmd_check,
                argparse.Namespace(base_url="", api_key="", timeout=to_int(timeout, 10), ua=ua),
            )
            pause_any_key()
            continue

        if choice == "4":
            validate = yes_no("是否开启 chat 可用性验证(--validate-chat)", default=True)
            prune_hint = "（添加场景无 prune）"
            print(f"validate-chat: {'开' if validate else '关'} {prune_hint}")
            timeout = input("超时秒数(默认10): ").strip() or "10"
            max_probe = input("max-probe(0=全量, 默认0): ").strip() or "0"
            ua = input(f"User-Agent(默认 {DEFAULT_UA}): ").strip() or DEFAULT_UA
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
            providers = input("要同步的 provider（序号/名称，空=全部）: ").strip()
            validate = yes_no("是否开启 chat 可用性验证(--validate-chat)", default=True)
            prune = False
            if validate:
                prune = yes_no("是否清理失效模型(--prune)", default=True)
            timeout = input("超时秒数(默认10): ").strip() or "10"
            max_probe = input("max-probe(0=全量, 默认0): ").strip() or "0"
            ua = input(f"User-Agent(默认 {DEFAULT_UA}): ").strip() or DEFAULT_UA
            no_restart = yes_no("仅改配置不重启 gateway(--no-restart)", default=False)
            run_action(
                cmd_sync,
                argparse.Namespace(
                    providers=providers,
                    validate_chat=validate,
                    prune=prune,
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
            provider = input("请输入 provider 序号或名称: ").strip()
            run_action(cmd_models, argparse.Namespace(provider=provider))
            pause_any_key()
            continue

        if choice == "8":
            names = input("要删除的 provider（支持序号/名称，空格分隔）: ").strip()
            no_restart = yes_no("仅改配置不重启 gateway(--no-restart)", default=False)
            run_action(cmd_remove, argparse.Namespace(name="", names=names, no_restart=no_restart))
            pause_any_key()
            continue

        if choice == "9":
            name = input("别名名称(默认 pm): ").strip() or "pm"
            path = input("安装路径(默认 /usr/local/bin 或 ~/bin): ").strip()
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
            "  pm ad --name demo1 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat\n"
            "  pm sy --providers demo2 --validate-chat --prune\n"
            "  pm sw --session\n"
            "\n"
            "提示:\n"
            "  pm 或 pm -h    进入数字菜单主界面\n"
            "  pm --cli-help  查看本页 CLI 帮助\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", aliases=["ck"], help="检测 API 是否可用（支持交互输入）")
    p_check.add_argument("--base-url", default="", help="例如: https://api.xxx.com/v1；不填则交互输入")
    p_check.add_argument("--api-key", default="", help="API Key；不填则交互输入")
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
            "  pm ad --name myp --base-url https://api.xxx.com/v1 --api-key <key>\n"
            "  pm ad --name demo2 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat\n"
            "  pm ad --name demo2 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat --max-probe 20\n"
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

    p_rm = sub.add_parser("remove", aliases=["rm"], help="删除 provider 并清理已注册模型（支持批量/序号）")
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
            "  pm sy --providers demo2 --validate-chat\n"
            "  pm sy --providers demo2 --validate-chat --prune\n"
            "  pm sy --providers 2 3 --validate-chat --timeout 12 --max-probe 50\n"
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

    p_list = sub.add_parser("list", aliases=["ls"], help="列出 provider 与注册状态（带序号）")
    p_list.set_defaults(func=cmd_list)

    p_models = sub.add_parser("models", aliases=["md"], help="列出某个 provider 的模型（带序号）")
    p_models.add_argument("--provider", required=True, help="provider 序号或名称")
    p_models.set_defaults(func=cmd_models)

    p_switch = sub.add_parser("switch", aliases=["sw"], help="切换默认模型：provider + 模型序号/名称")
    p_switch.add_argument("--provider", default="", help="provider 序号或名称；不填则交互选择")
    p_switch.add_argument("--model", default="", help="模型序号或模型 id，空则取该 provider 第一个")
    p_switch.add_argument("--session", action="store_true", help="同步切换当前会话模型（/model <provider/model>）")
    p_switch.add_argument("--no-restart", action="store_true", help="仅改配置，不重启 gateway")
    p_switch.set_defaults(func=cmd_switch)

    p_alias = sub.add_parser("alias-install", aliases=["ai"], help="安装命令别名（软链接），默认 pm")
    p_alias.add_argument("--name", default="pm", help="别名名称，默认 pm")
    p_alias.add_argument("--path", default="", help="可选：指定安装路径（默认为 /usr/local/bin，失败则 ~/bin）")
    p_alias.set_defaults(func=cmd_alias)

    return p


def main():
    # 交互主界面：pm -h 进入主菜单
    argv = sys.argv[1:]
    if not argv or "-h" in argv or "--help" in argv:
        return menu_loop()

    # 需要查看传统 CLI 帮助时使用 --cli-help
    if "--cli-help" in argv:
        argv = [x for x in argv if x != "--cli-help"] + ["--help"]

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except SystemExit as e:
        # argparse 的正常退出（例如 --help）
        return int(e.code) if isinstance(e.code, int) else 0
    except Exception as e:
        eprint(f"❌ 执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
