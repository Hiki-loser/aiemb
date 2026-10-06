# aiemb

Qualcomm QCS6490（广和通 SC171）嵌入式 AI 实验代码，包含模型导出、推理后端测试和离线智能家居语音助手。

## 目录

- 根目录的 Python 脚本：Qwen 模型导出、量化及 ONNX Runtime、QNN、Vulkan 推理实验。
- `hci_assistant/`：基于 Vosk、llama.cpp 和 espeak-ng 的离线语音助手。架构、运行与部署说明见 [hci_assistant/README.md](hci_assistant/README.md)。
- `toolchain-aarch64.cmake`：ARM64 交叉编译工具链配置。

## 本地资源

模型权重、ONNX 导出文件、Qualcomm SDK、安装包和 Python 虚拟环境体积较大，且部分资源受第三方许可约束，未纳入 Git。运行相关脚本前，请按脚本中的路径配置自行准备模型与 SDK。

设备连接配置和密钥仅保存在本地，不应提交到公开仓库。
