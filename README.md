# pm（OpenClaw Provider 管理脚本）

一个用于管理 OpenClaw Provider 与模型的工具，支持：
- 交互菜单模式（`pm -h`）
- 命令行子命令模式（`pm ck / ad / sy / sw / ls / md / rm / ai`）

---

## 功能概览

- 检测 API 连通性与模型可用性
- 添加 Provider 并注册模型
- 删除 Provider 并清理已注册模型
- 同步 Provider 模型到 `agents.defaults.models`
- 切换默认模型（可选同步当前会话）
- 安装/修复别名（例如 `pm`）
- 支持 `--validate-chat`：用 `chat/completions` 实测过滤不可用模型
- 支持 `--prune`：同步时清理已注册但不可用的模型

---

## 环境要求

- Linux
- Python 3.9+
- 已安装并可用 `openclaw` 命令
- 有权限读写 OpenClaw 配置

---

## 安装方式

### 方式一：手动安装到系统路径

```bash
sudo cp pm /usr/local/bin/pm
sudo chmod +x /usr/local/bin/pm
```

### 方式二：使用脚本内置别名安装

```bash
python3 pm ai --name pm
```

---

## 使用方法（重点）

### 1）菜单模式（推荐日常手动使用）

```bash
pm -h
```

说明：
- 输入数字编号进入功能
- 每次执行完可按任意键返回主界面
- `0` 退出

### 2）命令行模式（推荐自动化）

查看命令帮助：

```bash
pm --cli-help
```

常用命令：

```bash
# 检测 API
pm ck --base-url https://api.xxxx.com/v1 --api-key <key>

# 添加 provider，并验证 chat 可调用模型
pm ad --name demo1 --base-url https://api.xxxx.com/v1 --api-key <key> --validate-chat

# 同步模型；验证 chat；并清理已注册失效模型
pm sy --providers demo1 --validate-chat --prune

# 切换默认模型（交互）并同步当前会话模型
pm sw --session

# 列出 provider
pm ls

# 查看某个 provider 的模型（可填序号或名称）
pm md --provider 1

# 删除 provider（支持序号/名称，空格分隔）
pm rm --names 1 3 demo1

# 安装别名
pm ai --name pm
```

---

## 参数说明（高频）

- `--validate-chat`：通过 `POST /chat/completions` 验证模型是否真可用
- `--max-probe N`：限制验证模型数量；`0` 表示全量验证
- `--prune`：同步时清理已注册但验证失败的模型（通常与 `--validate-chat` 配合）
- `--session`：切换默认模型时，尝试同步当前会话模型
- `--no-restart`：只改配置，不重启 gateway

---

## 常见问题

### Q1：为什么 `/models` 能看到模型，但实际调用失败？
因为有些网关会返回“可见模型”，但该模型在 `chat/completions` 实际不可调用（404/403）。

建议：使用 `--validate-chat` 过滤。

### Q2：改了默认模型，为什么会话里还是旧模型？
默认模型与会话模型不是同一层。请使用：

```bash
pm sw --session
```

### Q3：菜单里文案怎么改？
编辑脚本中的 `lines = [...]` 文本即可；
- `__CENTER__:` 前缀表示居中
- `__SEP__` 表示分隔线

---

## 安全建议

- 不要把真实 API Key 提交到 GitHub
- 文档和示例请使用占位符（如 `demo1`、`https://api.xxxx.com/v1`、`<key>`）

---

## 建议目录结构

```text
.
├── pm
├── README.md
├── .gitignore
├── LICENSE
└── CHANGELOG.md
```
