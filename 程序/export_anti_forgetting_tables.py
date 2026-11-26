import argparse
import json
from pathlib import Path
from statistics import mean

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "程序/模型与日志/anti_forgetting/stage_metrics.json"
STAGES = ["initial_A", "after_B", "after_C", "after_D"]
TASKS = ["A_CWRU", "B_JNU", "C_UESTC", "D_Ottawa"]
LABELS = dict(zip(STAGES, ["初训 A", "训练 B 后", "训练 C 后", "训练 D 后"]))
METHOD_LABELS = {"sequential": "顺序微调基线", "ewc": "EWC", "pq_replay": "PQ 特征回放", "uciln": "UCILN 迁移"}
CLASSES = ["normal", "inner_race", "outer_race"]


def add_table(sheet, title, headers, rows, widths=None):
    if sheet.max_row > 1 or sheet["A1"].value:
        sheet.append([])
    sheet.append([title])
    sheet.merge_cells(start_row=sheet.max_row, start_column=1, end_row=sheet.max_row, end_column=len(headers))
    sheet.cell(sheet.max_row, 1).font = Font(bold=True, color="FFFFFF")
    sheet.cell(sheet.max_row, 1).fill = PatternFill("solid", fgColor="356185")
    sheet.append(headers)
    for cell in sheet[sheet.max_row]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5B9BD5")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for index, row in enumerate(rows):
        sheet.append(["—" if value is None else value for value in row])
        for cell in sheet[sheet.max_row]:
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            if index % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="EEF4F9")
    if widths:
        for letter, width in widths.items():
            sheet.column_dimensions[letter].width = width


def score_rows(result, method, tasks):
    rows = []
    for stage_index, stage in enumerate(STAGES[:len(tasks)]):
        drops, forgetting = [], []
        for task_index, task in enumerate(tasks):
            records = [seed_result["scores"][stage][task] for seed_result in result.values()]
            record = records[0]
            drop, forget = None, None
            if task_index < stage_index:
                histories = [
                    [seed_result["scores"][past][task]["accuracy"]
                     for past in STAGES[task_index:stage_index + 1]]
                    for seed_result in result.values()
                ]
                drop = 100 * mean(history[0] - history[-1] for history in histories)
                forget = 100 * mean(max(history[:-1]) - history[-1] for history in histories)
                drops.append(drop)
                forgetting.append(forget)
            rows.append([METHOD_LABELS[method], LABELS[stage], task, record["n"],
                         mean(item["accuracy"] for item in records),
                         mean(item["macro_recall"] for item in records),
                         mean(item["loss"] for item in records),
                         *(mean(item["per_class_recall"][name] for item in records) for name in CLASSES),
                         drop, forget, None, None, None])
        if drops:
            rows.append([METHOD_LABELS[method], LABELS[stage], "旧域汇总", None, None, None, None, None, None, None,
                         None, None, mean(drops), mean(forgetting), -mean(drops)])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--reports", type=Path, default=ROOT / "报告")
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    methods = list(payload)
    args.reports.mkdir(parents=True, exist_ok=True)
    for experiment, tasks in (("ABC", TASKS[:3]), ("ABCD", TASKS)):
        workbook = Workbook()
        notes = workbook.active
        notes.title = "指标说明"
        add_table(notes, "抗遗忘实验指标", ["指标", "计算口径", "解释"], [
            ("Accuracy", "正确窗口数 / 测试窗口数", "总体分类准确率。"),
            ("Macro Recall", "三个类别 Recall 的算术平均", "等权观察三类识别。"),
            ("Loss", "平均交叉熵 −log p(y|x)", "观察置信度和错误的严重程度。"),
            ("三类 Recall", "该类 TP / 该类真实样本数", "定位正常、内圈、外圈的类别退化。"),
            ("相对首次训练下降", "100 × (首次 Accuracy − 当前 Accuracy)", "正值表示相对初训发生下降。"),
            ("F_j(t)", "100 × (此前最佳 Accuracy − 当前 Accuracy)", "从该域首次受训起比较，不含当前阶段；负值表示超过此前最佳。"),
            ("AF", "旧域 F_j(t) 的均值", "当前阶段的平均遗忘，单位百分点。"),
            ("BWT", "100 × 旧域（当前 Accuracy − 首次 Accuracy）的均值", "单位百分点；负值代表负向后效。"),
            ("—", "不适用", "新域无历史参照；未来域不计入遗忘；AF、BWT 仅列在旧域汇总行。"),
        ], {"A": 24, "B": 48, "C": 64})
        add_table(notes, "方法和可比性", ["方法", "训练信息", "本实验定义"], [
            ("顺序微调基线", "当前域带标签训练", "复用原始顺序微调流程。"),
            ("EWC", "当前域带标签 + 对角 Fisher 参数约束", "Fisher 每域 128 个训练窗口，λ=1000。"),
            ("PQ 特征回放", "仅 A 域有标签特征；不学习当前域", "冻结 WDCNN 特征提取器，只训练分类头；不做 MMD。"),
            ("UCILN 迁移", "A 域有标签，其余域仅输入", "按论文 Algorithm 1 的 PQ 记忆、特征回放和 MMD 近似迁移；四域实验保留三类头。"),
            ("协议限制", "所有方法都用同一 A 初训检查点和测试切分", "UCILN 与 EWC 的标签可用性不同，结果为探索性对照，不是严格同预算排行榜。"),
        ], {"A": 22, "B": 48, "C": 70})
        sheet = workbook.create_sheet("实验记录")
        headers = ["方法", "阶段", "测试域", "测试窗口数", "Accuracy", "Macro Recall", "Loss", "正常 Recall", "内圈 Recall", "外圈 Recall", "相对首次下降(pp)", "F_j(pp)", "旧域下降均值(pp)", "AF(pp)", "BWT(pp)"]
        all_rows = []
        for method in methods:
            all_rows.extend(score_rows(payload[method], method, tasks))
        add_table(sheet, f"{experiment} 抗遗忘方法对比（种子均值）", headers, all_rows,
                  {"A": 18, "B": 14, "C": 16, "D": 12, "E": 12, "F": 14, "G": 12,
                   "H": 14, "I": 14, "J": 14, "K": 18, "L": 12, "M": 18, "N": 12, "O": 12})
        for row in sheet.iter_rows():
            for cell in row:
                if cell.column in {5, 6, 8, 9, 10} and isinstance(cell.value, float):
                    cell.number_format = "0.00%"
                if cell.column == 7 and isinstance(cell.value, float):
                    cell.number_format = "0.0000"
                if cell.column >= 11 and isinstance(cell.value, (int, float)):
                    cell.number_format = "0.00"
        seeds_sheet = workbook.create_sheet("A末端逐种子")
        rows = []
        endpoint = STAGES[len(tasks) - 1]
        for method in methods:
            for seed, result in payload[method].items():
                scores = result["scores"]
                start = scores["initial_A"]["A_CWRU"]["accuracy"]
                end = scores[endpoint]["A_CWRU"]["accuracy"]
                rows.append([METHOD_LABELS[method], int(seed), start, end, 100 * (start - end),
                             scores[endpoint]["A_CWRU"]["macro_recall"], scores[endpoint]["A_CWRU"]["loss"]])
        add_table(seeds_sheet, f"{experiment} A 域末端逐种子结果", ["方法", "随机种子", "初训 A Accuracy", "末端 A Accuracy", "下降(pp)", "末端 Macro Recall", "末端 Loss"], rows,
                  {"A": 18, "B": 12, "C": 18, "D": 18, "E": 12, "F": 18, "G": 14})
        for row in seeds_sheet.iter_rows():
            for cell in row:
                if cell.column in {3, 4, 6} and isinstance(cell.value, float):
                    cell.number_format = "0.00%"
        workbook.save(args.reports / f"{experiment}抗遗忘方法对比.xlsx")


if __name__ == "__main__":
    main()
