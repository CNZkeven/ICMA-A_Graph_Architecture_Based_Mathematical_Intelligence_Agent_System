import argparse  # 导入命令行参数解析库
import asyncio  # 导入异步编程库
import json  # 导入JSON处理库
import os  # 导入操作系统接口库
from pathlib import Path  # 导入路径操作库
from typing import Dict, List  # 导入类型提示库

from llm_client import InternChatClient  # 导入大语言模型客户端
from user_agent import ReasoningAgent  # 导入推理代理类


LOCAL_MAX_CONCURRENCY = int(os.environ.get("LOCAL_MAX_CONCURRENCY", "8"))  # 从环境变量获取最大并发数，默认为8


def load_jsonl(path: Path) -> List[Dict]:  # 定义加载JSONL文件函数
    items = []  # 初始化项目列表
    with path.open("r", encoding="utf-8") as file:  # 以UTF-8编码打开文件
        for line_number, line in enumerate(file):  # 遍历文件每一行并记录行号
            if not line.strip():  # 如果行为空白则跳过
                continue
            item = json.loads(line)  # 解析JSON行数据
            item.setdefault("idx", line_number)  # 设置默认索引为行号
            items.append(item)  # 将项目添加到列表
    return items  # 返回项目列表


def result_path(output_dir: Path, item: Dict) -> Path:  # 定义结果文件路径生成函数
    return output_dir / f"{item['idx']}.json"  # 返回输出目录下以索引命名的JSON文件路径


def is_processed(path: Path) -> bool:  # 定义检查是否已处理函数
    return path.exists() and path.stat().st_size > 0  # 检查文件是否存在且大小大于0


def write_json(path: Path, record: Dict) -> None:  # 定义写入JSON文件函数
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如果不存在）
    tmp_path = path.with_suffix(path.suffix + ".tmp")  # 创建临时文件路径
    with tmp_path.open("w", encoding="utf-8") as file:  # 以UTF-8编码打开临时文件
        json.dump(record, file, ensure_ascii=False, indent=2)  # 将记录写入文件，不转义ASCII字符，缩进2格
        file.write("\n")  # 写入换行符
    tmp_path.replace(path)  # 将临时文件替换为正式文件（原子操作）


def build_output_record(item: Dict, agent_result: Dict) -> Dict:  # 定义构建输出记录函数
    final_response = agent_result.get("final_response", "")  # 获取最终响应内容
    if not isinstance(final_response, str) or not final_response.strip():  # 检查响应是否为非空字符串
        raise ValueError("agent.solve must return a non-empty string field: final_response")  # 抛出值错误异常

    output = {  # 构建输出字典
        "idx": item["idx"],  # 索引
        "status": "success",  # 处理状态
        "final_response": final_response,  # 最终响应
        "trace": agent_result.get("trace", []),  # 跟踪信息
    }
    return output  # 返回输出字典


def parse_args() -> argparse.Namespace:  # 定义解析命令行参数函数
    parser = argparse.ArgumentParser(description="Competition sample reasoning agent.")  # 创建参数解析器
    parser.add_argument("--input_file", required=True, help="Path to input JSONL.")  # 添加输入文件参数
    parser.add_argument("--output_dir", required=True, help="Directory for per-problem JSON outputs.")  # 添加输出目录参数
    return parser.parse_args()  # 返回解析后的参数


def solve_item(agent: ReasoningAgent, item: Dict) -> Dict:  # 定义解决单个项目函数
    result = agent.solve(  # 调用代理解决方法
        problem=item["problem"],  # 传入问题内容
        metadata={"idx": item["idx"]},  # 传入元数据（索引）
    )
    return build_output_record(item, result)  # 构建并返回输出记录


async def process_item(  # 定义异步处理单个项目函数
    agent: ReasoningAgent,  # 推理代理对象
    item: Dict,  # 待处理项目
    output_dir: Path,  # 输出目录
    semaphore: asyncio.Semaphore,  # 并发控制信号量
) -> None:
    path = result_path(output_dir, item)  # 获取结果文件路径
    if is_processed(path):  # 检查是否已处理
        print(f"Skip idx={item['idx']} because {path} already exists.")  # 打印跳过信息
        return  # 返回

    async with semaphore:  # 异步获取信号量（控制并发）
        try:
            record = await asyncio.to_thread(solve_item, agent, item)  # 在线程池中执行同步函数
        except Exception as exc:  # 捕获所有异常
            record = {  # 构建错误记录
                "idx": item["idx"],  # 索引
                "status": "error",  # 错误状态
                "final_response": "",  # 空响应
                "error": {  # 错误详情
                    "type": type(exc).__name__,  # 错误类型名
                    "message": str(exc),  # 错误消息
                },
                "trace": [],  # 空跟踪信息
            }
        await asyncio.to_thread(write_json, path, record)  # 在线程池中写入JSON文件
        print(f"Finished idx={item['idx']}")  # 打印完成信息


async def run(args: argparse.Namespace) -> None:  # 定义运行主函数
    input_path = Path(args.input_file)  # 创建输入路径对象
    output_dir = Path(args.output_dir)  # 创建输出目录对象

    items = load_jsonl(input_path)  # 加载JSONL文件中的项目

    client = InternChatClient()  # 创建大语言模型客户端
    agent = ReasoningAgent(client=client)  # 创建推理代理
    semaphore = asyncio.Semaphore(LOCAL_MAX_CONCURRENCY)  # 创建并发控制信号量

    print(f"Loaded {len(items)} items. Max concurrency: {LOCAL_MAX_CONCURRENCY}.")  # 打印加载信息
    tasks = [process_item(agent, item, output_dir, semaphore) for item in items]  # 创建任务列表
    await asyncio.gather(*tasks)  # 并发执行所有任务
    print(f"Saved outputs to {output_dir}")  # 打印保存完成信息


def main() -> None:  # 定义主函数
    asyncio.run(run(parse_args()))  # 运行异步主程序


if __name__ == "__main__":  # 当作为主脚本运行时
    main()  # 执行主函数


# 此代码是一个基于大语言模型的推理代理批量处理系统，支持异步并发处理JSONL格式的输入文件，
# 对每个问题项进行推理求解，并将结果保存为单独的JSON文件，具有错误处理和进度跟踪功能。
# 支持断点续传（跳过已处理文件）和并发控制，适用于大规模推理任务的自动化处理。