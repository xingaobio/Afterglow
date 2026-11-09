# Afterglow systemd / udev 单元

| 文件 | 作用 | 默认启用？ |
|---|---|---|
| `afterglow-llm.service` | llama-server，视觉模型 Qwen3-VL-4B，`127.0.0.1:8080` | 是（install.sh 在模型下载完成后 `enable --now`） |
| `afterglow-brain.service` | llama-server，可选 Qwen3.6-35B-A3B（`--n-cpu-moe 99`），`127.0.0.1:8081`，**与 llm 互斥**（`Conflicts=`） | 否，按需启动 |
| `afterglow-watch.service` | `afterglow watch`，挂载出现/定时触发增量运行 | 否（`install.sh --enable-watch` 或手动 enable） |
| `afterglow-nightly.service` + `.timer` | 每天 03:17 `afterglow run`（`Persistent=true`，错过会补跑） | timer 是 |
| `afterglow-run@.service` | 模板：某个卷标的盘出现并挂载后跑一次 `afterglow run` | 由 udev 触发 |
| `99-afterglow-usb.rules` | udev 规则：USB 文件系统 `add` → `systemctl --no-block start afterglow-run@<卷标>.service` | 随 install.sh 安装 |

## 需要按你的机器修改的地方

install.sh 会把单元里的 `@AFTERGLOW_USER@` 替换成运行 install.sh 的用户并复制到 `/etc/systemd/system/`。
手动安装时请自己替换：

```bash
for f in afterglow-*.service afterglow-*.timer; do
  sed 's/@AFTERGLOW_USER@/你的用户名/' "$f" | sudo tee "/etc/systemd/system/$f" >/dev/null
done
sudo cp 99-afterglow-usb.rules /etc/udev/rules.d/
sudo systemctl daemon-reload && sudo udevadm control --reload
```

需要核对/修改的路径（都写死在单元的 `ExecStart=` 里，默认值与 `deploy/` 下其它脚本一致）：

- **用户**：`User=@AFTERGLOW_USER@`。该用户必须能读写 `/data/afterglow`、能读 `paths.source`。配置文件取 `~/.config/afterglow/afterglow.toml`（单元没有传 `--config`）。
- **Afterglow 可执行文件**：`/opt/afterglow/.venv/bin/afterglow`（install.sh 第 6 步创建的 venv）。
- **llama-server**：`/opt/llama.cpp/build/bin/llama-server`。新版 llama.cpp 若改了二进制名（部分模型卡已出现 `llama serve`），请改 `ExecStart`。
- **模型文件**（`/data/afterglow/models/gguf/`，文件名见 `deploy/models/models.lock`）：
  - llm：`Qwen3VL-4B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen3VL-4B-Instruct-F16.gguf`
  - brain：`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` + `mmproj-Qwen3.6-35B-A3B-F16.gguf`
- **`--alias`**：必须与 `afterglow.toml` 里 `[llm].vision_model`（默认 `qwen3-vl-4b`）/ `brain_model`（默认 `qwen3.6-35b-a3b`）一致，否则 API 请求里的模型名对不上。
- **端口**：8080 / 8081 要与 `[llm].server` / `[llm].brain_server` 一致。
- **udev 规则**：默认规则 B 对**任何带卷标的 USB 文件系统**触发；只想对你的移动硬盘触发，请改用文件里注释着的规则 A 并写上卷标。规则使用 `ID_FS_LABEL_ENC`（空格会被转成 `\x20`），否则含空格的卷标不是合法的 systemd 实例名。
- **GPU 互斥**：llm / brain 各占一部分显存，8 GB 放不下两个，所以 `Conflicts=`。`afterglow run` 里 GPU 阶段另有 `gpu.lock_file` 文件锁，和 Immich 的 ML 任务也互斥（见 ARCHITECTURE.md §6 `gpu_lock.py`）。

## 常用命令

```bash
systemctl status afterglow-llm
journalctl -u afterglow-llm -f                       # 模型加载与请求日志
systemctl list-timers afterglow-nightly.timer
sudo systemctl start afterglow-brain                 # 自动停掉 afterglow-llm
sudo systemctl start afterglow-llm                   # 切回视觉模型（自动停掉 brain）
sudo systemctl start 'afterglow-run@PHOTOS.service'  # 手动模拟"插盘"
```

## 让 afterglow（非 root）切换 llm / brain（可选，本仓库未实际测试）

ARCHITECTURE.md 约定由 afterglow 在需要事件命名时 `systemctl start afterglow-brain`。普通用户默认没有权限，
可加一条 polkit 规则（把 `xin` 换成你的用户名），放到 `/etc/polkit-1/rules.d/50-afterglow.rules`：

```js
polkit.addRule(function(action, subject) {
  if (action.id == "org.freedesktop.systemd1.manage-units" &&
      subject.user == "xin" &&
      action.lookup("unit").indexOf("afterglow-") == 0) {
    return polkit.Result.YES;
  }
});
```

没有这条规则时，`brain_server` 留空即可：事件命名会改用视觉模型。

## 卸载

```bash
sudo systemctl disable --now afterglow-nightly.timer afterglow-watch afterglow-llm afterglow-brain
sudo rm /etc/systemd/system/afterglow-* /etc/udev/rules.d/99-afterglow-usb.rules
sudo systemctl daemon-reload && sudo udevadm control --reload
```
