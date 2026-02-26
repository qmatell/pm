# openclaw manager 使用说明（README）
下载脚本
```bash
curl -L -o opm.py "https://raw.githubusercontent.com/qmatell/opm/refs/heads/main/opm.py"
```
./opm.py就行
## ⚡ 最短速查（10 秒上手）

```bash
# 1) 进入数字菜单（推荐）
./opm.py
如果没有权限就chmod +x ./opm.py
进入后使用9创建别名，默认是opm，创建别名后任何地方opm都能唤出

# 2) 看 CLI 总帮助
./opm.py --cli-help
# 或（已装别名时）
opm --cli-help

# 3) 常用命令
opm list                                  # 列 provider
opm check                                 # 检测本地全部 API
opm check --providers "1 2"               # 检测指定 provider
opm check --new-api --base-url ... --api-key ...
opm add --name demo1 --base-url ... --api-key ...
opm models --provider "1 2"               # 查看并检测模型可用性
opm sync --providers "1 2"                # 同步可用模型并清理失效
opm switch --provider 1 --model 2          # 切换默认模型

# 4) OpenClaw 运维
opm oc-start | opm oc-stop | opm oc-restart | opm oc-status
opm oc-check-update | opm oc-doctor | opm oc-uninstall
```

> 返回/退出统一用 `q`（菜单模式）。

> 文件位置：`./opm.py`
>
> 本脚本用于 OpenClaw 的 provider 管理、模型可用性检测、同步注册，以及 OpenClaw/Gateway 运维。

---

## 一、脚本能力总览

脚本提供两种使用模式：

1. **数字菜单模式**（推荐日常使用）
   - 直接运行：`python3 opm.py`
2. **命令行模式（CLI）**
   - 例如：`python3 opm.py check ...`
   - 或你已安装别名后：`opm check ...`

---

## 二、数字菜单模式

启动方式：

```bash
python3 ./opm.py
```

主菜单功能：

- `1. openclaw设置`
- `2. 列出 provider（list）`
- `3. 检测 API 可用性（check）`
- `4. 添加 provider（add）`
- `5. 同步模型注册（sync）`
- `6. 切换默认模型（switch）`
- `7. 查看 provider 模型（models）`
- `8. 删除 provider（remove）`
- `9. 安装/修复别名（alias-install）`
- `00. 更新/卸载（本脚本）`
- `q. 退出`

> 菜单内返回/退出统一使用 `q`。

---

## 三、openclaw设置（菜单1）

子菜单：

1. 安装 OpenClaw
2. 启动 gateway
3. 停止 gateway
4. 重启 gateway
5. 状态日志查看
6. 检测 OpenClaw 更新
7. 卸载 OpenClaw
8. 健康检测/修复（doctor）
q. 返回主菜单

### 说明

- **重启**已做增强：会先清理 `tmux gateway` 会话，再 stop/start，减少配置未刷新问题。
- **状态日志查看**会输出状态与日志，便于定位问题。

---

## 四、命令行模式（CLI）

查看总帮助：

```bash
python3 ./opm.py --cli-help
# 或
opm --cli-help
```

查看某个子命令帮助：

```bash
opm <子命令> -h
```

例如：

```bash
opm check -h
opm models -h
opm sync -h
```

---

## 五、核心功能说明

### 1）check：检测 API 可用性

支持两类检测：

- **检测本地已添加 provider（默认）**
  - 可指定 provider（序号/名称，支持多个）
  - 输出 ✅/❌ 逐条结果
  - 检测完成后汇总：可用数量、不可用数量、不可用序号
- **检测新 API**（临时 base-url/api-key）
  - 适合添加前先验通

示例：

```bash
# 默认检测本地全部 provider API
opm check

# 只检测指定 provider
opm check --providers "1 2"
opm check --providers demo1 demo2

# 检测新 API
opm check --new-api --base-url https://api.xxx.com/v1 --api-key sk-xxxx
```

---

### 2）add：添加 provider

行为要点：

- 先调用 `/models` 拉取模型
- **再逐个做 chat 可用性检测**（实时打印进度）
- **只注册可用模型**（不可用模型会过滤）
- 写入：
  - `models.providers.<provider>`
  - `agents.defaults.models`

示例：

```bash
opm add --name demo1 --base-url https://api.xxx.com/v1 --api-key sk-xxxx
```

---

### 3）models：查看 provider 模型

支持：

- 单个或多个 provider（如 `"1 2"`）
- 默认检测模型可用性，逐条打印 ✅/❌
- 可关闭检测（仅列模型）

示例：

```bash
opm models --provider demo1
opm models --provider "1 2"
opm models --provider "demo1 demo2" --no-check
```

---

### 4）sync：同步模型注册

当前逻辑：

- 默认会检测 provider 下模型可用性
- **只同步可用模型**
- 对于已注册但现在不可用/失效的模型，会自动移除
- 输出每个 provider 的统计信息，例如：
  - provider模型数
  - 可用数
  - 不可用数
  - 新增数
  - 移除数
  - 已注册数

示例：

```bash
# 全量同步
opm sync

# 仅同步指定 provider
opm sync --providers "1 2"

# 控制探测参数
opm sync --providers demo1 --timeout 12 --max-probe 50
```

---

### 5）switch：切换默认模型

示例：

```bash
opm switch --provider demo1 --model gpt-4o-mini
# 或按序号
opm switch --provider 1 --model 2
```

可选 `--session`：尝试同步切换当前会话模型。

---

### 6）remove：删除 provider

会删除 provider 并清理其已注册模型。

示例：

```bash
opm remove --name demo1
opm remove --names "1 3 demo2"
```

---

### 7）alias-install：安装命令别名

示例：

```bash
opm alias-install --name opm
```

默认优先 `/usr/local/bin`，失败则尝试 `~/bin`。

---

## 六、openclaw 运维命令（CLI）

与菜单1对应的非交互命令：

```bash
opm oc-install
opm oc-start
opm oc-stop
opm oc-restart
opm oc-status
opm oc-check-update
opm oc-uninstall
opm oc-doctor
```

---

## 七、本脚本自更新/卸载（菜单00）

功能：

- 检测 GitHub 是否有更新
- 更新当前脚本
- 卸载当前脚本并删除 

备注：

- 更新前会备份旧脚本到同目录下，且仅保留一份覆盖备份。

---

## 八、常见问题

### Q1：为什么“重启”后 Telegram `/models` 没马上更新？

可能是旧 gateway 会话未彻底释放。当前脚本已把重启优化为“清理会话 + stop/start + 状态探测”，与手工停启效果对齐。

### Q2：add 会把不可用模型注册进去吗？

不会。当前 add 已改为只注册检测可用模型。

### Q3：sync 会自动清理已失效模型吗？

会。sync 会移除不可用/失效模型，并输出移除数量。

---

## 九、建议使用方式

- 日常用菜单：更直观
- 自动化/批处理用 CLI：更稳定可复用
- 变更后建议看一次 `oc-status` 确认状态

---

如需扩展（比如导出检测报告、并发检测、失败重试策略），可以继续在此 README 下追加“高级用法”。
