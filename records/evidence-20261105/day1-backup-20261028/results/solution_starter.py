"""学生作品入口：请在本文件中实现自己的 ProGen2-medium 推理方案。

本文件仅定义接口，不包含模型加载、生成或优化实现。先阅读
docs/INTERFACE.md，再按约定实现 Solution。统一 FP32 对照放在 baseline/ 中。
"""

from __future__ import annotations


class Solution:
    """评测程序通过这个类加载并调用学生作品。"""

    def __init__(self, model_dir: str, tokenizer_path: str, device: str = "cuda:0"):
        """初始化模型及输入无关资源，初始化不计入推理时间。

        Args:
            model_dir: 评测程序传入的 ProGen2-medium 权重与配置目录。
            tokenizer_path: 评测程序传入的 tokenizer.json 文件路径。
            device: 分配的计算设备；DCU 版 PyTorch 仍使用 cuda:0 等名称。

        请自行补充模型加载与初始化代码，使用传入路径，不要写死个人路径。
        """
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.device = device

    def reset(self) -> None:
        """可选：清除上一请求的 KV Cache、生成结果及其他输入相关状态。

        若实现从不跨请求保留这些状态，本方法可保持为空。可以保留模型、
        已编译算子、输入无关的 RoPE 表和可复用缓冲区。
        """
        pass

    def generate(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        """按输入顺序返回新生成的氨基酸后缀，不包含原输入前缀。

        Args:
            prefixes: 同一请求的输入前缀列表，列表长度即本次输入条数。
            max_new_tokens: 每条结果必须生成的氨基酸字母数。

        Returns:
            与 prefixes 等长的 list[str]，每个字符串恰含 max_new_tokens 个
            大写氨基酸字母，字符范围为 ACDEFGHIKLMNPQRSTVWY。

        生成规则、精度与结果要求见赛题说明及 docs/INTERFACE.md。
        """
        raise NotImplementedError(
            "submission/solution.py 只是接口骨架，请先实现模型加载和 generate()，"
            "再运行作品测试。查看统一 FP32 对照请运行 bash run_baseline.sh。"
        )
