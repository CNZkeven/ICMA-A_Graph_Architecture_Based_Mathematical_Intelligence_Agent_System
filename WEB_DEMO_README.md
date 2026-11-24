# ICMA Web Demo

前端位于 `web/`，本机适配层位于 `demo_server/`。推荐使用 Conda 的
`competition` 环境运行，真实求解的 Key 只通过运行请求头传给后端，不写入
前端构建产物。

## 启动

Windows 用户可以直接双击项目根目录的 `启动Demo.cmd`。脚本会自动完成以下工作：

- 优先使用名为 `competition` 的 Conda 环境；如果本机没有该环境，则用系统自带的
  Python 3.10+ 创建项目内 `.venv`。
- 缺少依赖时自动安装 `requirements.txt`，并按需安装前端依赖。
- 前端源码有更新时自动执行生产构建。
- 启动 FastAPI 服务，通过健康检查后自动用默认浏览器打开 Demo。
- 如果 Demo 已在运行，则直接打开现有页面；默认端口被其他程序占用时，会自动选择附近空闲端口。

双击 `停止Demo.cmd` 可以停止由启动脚本创建的 Demo 服务及其子进程。首次启动安装依赖可能
需要数分钟，服务日志位于 `logs/demo-server.out.log` 和 `logs/demo-server.err.log`。

也可以在 PowerShell 中使用参数启动：

```powershell
.\scripts\deploy_demo.ps1 -Port 8765
.\scripts\deploy_demo.ps1 -NoBrowser -ForceInstall -Rebuild
```

手动启动方式：

```powershell
conda activate competition
cd C:\Users\keven\Desktop\TZB\ICMAnew
python -m uvicorn demo_server.app:app --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765/`。如果需要重新构建前端：

```powershell
cd web
npm ci
npm run build
```

左下角 `Key` 菜单用于设置模型 Key。“完整智能体”默认开启，自由输入运行完整 ICMA；
关闭后使用同一模型和 Key，携带当前聊天历史直接流式回答，模型自身的思考模式仍开启
（`thinking_mode=true`）。“完整智能体”开关控制是否使用 ICMA 框架。开关选择会在浏览器中保留。
直接对话的模型思考内容会在答案上方单独流式展示，支持折叠、公式渲染和刷新恢复；
中断时保留已收到的思考与回答。系统提示词明确 ICMA 数学智能体身份、数学解题职责和实际工具能力。
三道基础示例默认播放阶段回放；“真实重算”按开关选择执行模式。两道竞赛题始终
播放录制过程，不受开关影响。超长公式可以在公式区域内横向滚动。
历史消息与运行事件保存在 `demo_data/icma-demo.sqlite3`。

输入框右侧固定显示“停止生成”按钮，空闲时灰显；对话、ICMA 求解和示例回放运行时
按钮变为红色。点击后会强制
结束当前运行及其 Python 执行子进程，保留已生成的回答、模型思考与阶段轨迹，并标记
“已停止”；可以继续发送新消息，刷新后也能查看保留下来的内容。每轮运行使用独立进程，
无需等待模型的网络超时。服务关闭时会停止并保存仍在进行的运行。

`INTERN_API_BASE` 可以填写基础地址 `https://chat.intern-ai.org.cn/api/v1`，也可以填写
完整的 `/chat/completions` 地址，客户端会自动补齐路径。普通流式对话的单次建连上限为
10 秒，等待模型输出仍保留 280 秒；尚未输出内容的网络故障最多重试一次，已有输出后
不会自动重发。完整 ICMA 的调用重试次数保持原有设置。

## 临时公网访问（无需账号或域名）

安装 `cloudflared` 并启动本地 Demo 后，双击根目录的 `开启公网访问.cmd`。
脚本读取本地 Demo 的实际端口，通过 Cloudflare 临时隧道生成
`https://*.trycloudflare.com` 地址，并自动打开浏览器。把该地址发给访问者即可。
脚本也会在新 PowerShell 尚未识别 PATH 时查找 WinGet 的用户安装位置。

临时隧道下，前端自动使用按序号读取的 JSON 事件轮询，持续更新回答、思考过程和阶段状态，
避开临时隧道的 SSE 限制。本地默认继续使用 SSE；SSE 连接出错时也会切换到事件轮询。
其他代理环境可在页面地址后加 `?transport=poll` 强制使用轮询。

双击 `停止公网访问.cmd` 可关闭隧道，本地 Demo 继续运行。再次开启时地址可能变化。
电脑需要保持开机、联网并避免休眠。隧道日志位于 `logs/cloudflared.err.log`，
运行信息和当前公网地址保存在 `.demo-runtime/tunnel.json`。

当前版本没有登录认证或访客会话隔离，访客可以查看和操作同一批历史对话；
若后端配置了模型 Key，访客也可能消耗该 Key 的额度。适合向可信人员临时演示。
从本地地址切换到公网地址后，浏览器保存的 Key 不会自动迁移，真实求解可在公网页面重新填写。

## 验证

```powershell
conda activate competition
python -m pytest tests/test_demo_server.py tests/test_llm_client.py -q
cd web
npm run build
```
