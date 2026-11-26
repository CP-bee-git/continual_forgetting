import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from run_cross_dataset_sequence import OUTPUT, ROOT, TASKS


STAGES = ["initial_A", "after_B", "after_C", "after_D"]
STAGE_LABELS = dict(zip(STAGES, ["初训 A", "训练 B 后", "训练 C 后", "训练 D 后"]))
HEADERS = ["阶段", "测试域", "窗口数", "Accuracy", "Macro Recall", "Loss", "正常 Recall",
           "内圈 Recall", "外圈 Recall", "相对首次训练下降 (pp)", "F_j(t) (pp)", "AF (pp)", "BWT (pp)"]
METRICS = [
    ("Accuracy", "正确预测窗口数 / 测试窗口总数", "总体分类正确率，用于计算遗忘量。"),
    ("Macro Recall", "三个类别 Recall 的算术平均", "等权观察各类别，减少类别比例差异对解读的影响。"),
    ("Loss", "−Σ log p(真实类别) / 测试窗口总数", "平均交叉熵，用于观察模型对真实标签的置信程度。"),
    ("正常、内圈、外圈 Recall", "该类正确预测窗口数 / 该类真实窗口数", "定位哪类标签出现退化。"),
    ("相对首次训练下降", "R[j,j] − R[t,j]", "正值表示该域相对首次学完时的 Accuracy 下降。"),
    ("F_j(t)", "max(R[k,j]，j ≤ k < t) − R[t,j]", "相对该域在当前阶段之前的历史最佳 Accuracy 的下降。"),
    ("AF", "平均 F_j(t)，仅纳入 j < t", "当前阶段对已学旧域的平均遗忘。"),
    ("BWT", "平均 (R[t,j] − R[j,j])，仅纳入 j < t", "负值表示当前阶段对旧域整体产生负向后效。"),
]


def metrics_rows(result, tasks, experiment):
    scores = result["scores"]
    stages = STAGES[:len(tasks)]
    rows = []
    for index, stage in enumerate([*stages, f"relearn_{experiment}"]):
        label = STAGE_LABELS.get(stage, "A 再学习后")
        drops, forgetting = [], []
        for position, task in enumerate(tasks):
            record = scores[stage][task]
            rec = record["per_class_recall"]
            drop, forget = None, None
            if position < index < len(tasks):
                accuracy = record["accuracy"]
                drop = 100 * (scores[stages[position]][task]["accuracy"] - accuracy)
                forget = 100 * (max(scores[previous][task]["accuracy"]
                                    for previous in stages[position:index]) - accuracy)
                drops.append(drop)
                forgetting.append(forget)
            rows.append([label, task, record["n"], record["accuracy"], record["macro_recall"],
                         record["loss"], rec["normal"], rec["inner_race"], rec["outer_race"],
                         drop, forget, None, None])
        if drops:
            rows.append([label, "旧域汇总", *([None] * 9), mean(forgetting), -mean(drops)])
    return rows


def style_row(sheet, row, color):
    for cell in sheet[row]:
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def table(sheet, title, headers, rows, formats):
    if sheet["A1"].value is not None:
        sheet.append([None])
    sheet.append([title])
    sheet.merge_cells(start_row=sheet.max_row, start_column=1, end_row=sheet.max_row, end_column=len(headers))
    style_row(sheet, sheet.max_row, "17365A")
    sheet.row_dimensions[sheet.max_row].height = 23
    sheet.append(headers)
    style_row(sheet, sheet.max_row, "356185")
    sheet.row_dimensions[sheet.max_row].height = 32
    for index, row in enumerate(rows):
        sheet.append(["—" if item is None else item for item in row])
        for cell in sheet[sheet.max_row]:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if cell.column in formats:
                cell.number_format = formats[cell.column]
            if index % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="EEF4F9")


def export(experiment, results, config, manifests, reports):
    tasks = TASKS[:len(experiment)]
    endpoint = STAGES[len(tasks) - 1]
    relearn = f"relearn_{experiment}"
    seeds = list(results)
    workbook = Workbook()
    notes = workbook.active
    notes.title = "指标说明"
    table(notes, "指标与计算口径", ["指标", "简要公式", "含义"], METRICS, {})
    table(notes, "实验取舍", ["项目", "设置", "解释"], [
        ("任务序列", " → ".join(tasks), "ABC 和 ABCD 共用 A→B→C 检查点，每条序列单独接 A 再学习。"),
        ("A 可切换配置", "12 kHz 原始与目标采样率，2048 点窗口，步长 512，逐窗 z-score，正常/内圈/外圈映射为 0/1/2。",
         "域专属配置包含初训 A 锁定的 BatchNorm 运行均值、方差和计数。各域使用共享三分类头，无适配器。"),
        ("共享模型状态", "WDCNN 卷积、分类头与 BatchNorm 可训练仿射参数全量微调。", "阶段末保留最后一次更新后的权重。"),
        ("无更新回访", "切换输入、标签与首次学完时的域 BatchNorm 统计，启用 eval 和 inference_mode。",
         "权重始终来自当前阶段。主表已将该方式用于所有已学域，首次训练前使用当前共享统计。"),
        ("A 再学习", "从序列末端权重和初训 A 统计出发，用 A 的训练分割再跑 10 轮。", "评估 A 使用再学习后的统计。B、C、D 仍使用各自首次学完后的统计。"),
        ("更新预算", f"主序列每域 {config['steps_per_domain']} 步，每步 {config['batch_size']} 个窗口，Adam，lr={config['lr']}。",
         "尾批丢弃会累加重复遍历的计数。预算未取用指末次部分遍历未纳入批次的窗口出现次数。"),
        ("测试切分", "A/B/C 按原始时间 70/15/15 切分后各自预处理。D 按整条记录分割，试次 1/2/3 分别用于训练/验证/测试。",
         "A/B/C 结果只反映所选记录内时段泛化。D 为记录级留出。"),
        ("重复与汇总", f"随机种子 {', '.join(seeds)}，测试集复用，先逐种子算遗忘再平均。",
         "波动以种子间样本标准差描述。pp 为百分点，— 表示不适用。"),
    ], {})
    for column, width in zip("ABC", (26, 67, 75)):
        notes.column_dimensions[column].width = width
    for row in range(3, notes.max_row + 1):
        notes.row_dimensions[row].height = 33

    sheet = workbook.create_sheet("实验记录")
    raw_rows = {seed: metrics_rows(results[seed], tasks, experiment) for seed in seeds}
    summary = []
    for repeated_rows in zip(*raw_rows.values()):
        base = repeated_rows[0]
        summary.append([*base[:3], *(None if base[col] is None else mean(row[col] for row in repeated_rows)
                                       for col in range(3, len(HEADERS)))])
    score_formats = {4: "0.00%", 5: "0.00%", 6: "0.0000", 7: "0.00%", 8: "0.00%", 9: "0.00%",
                     10: "0.00", 11: "0.00", 12: "0.00", 13: "0.00"}
    table(sheet, f"{experiment} 各阶段固定测试集结果（种子均值）", HEADERS, summary, score_formats)

    primary = []
    for seed, result in results.items():
        scores = result["scores"]
        start, revisit, recovered = (scores[stage][TASKS[0]]["accuracy"] for stage in (STAGES[0], endpoint, relearn))
        shared = result["a_with_shared_bn"][endpoint]["accuracy"]
        primary.append([seed, start, shared, revisit, 100 * (start - revisit), recovered,
                        100 * (recovered - revisit), 100 * (recovered - start)])
    primary += [["均值", *(mean(column) for column in zip(*(row[1:] for row in primary)))]]
    if len(seeds) > 1:
        primary.append(["标准差", *(stdev(column) for column in zip(*(row[1:] for row in primary[:-1])))])
    table(sheet, "A 回访和 10 轮再学习", ["种子", "初训 A", "末端共享 BN", "切回 A，无更新", "A 下降 (pp)",
          "A 再学习后", "相对回访提升 (pp)", "相对初训变化 (pp)"], primary,
          {2: "0.00%", 3: "0.00%", 4: "0.00%", 5: "0.00", 6: "0.00%", 7: "0.00", 8: "0.00"})

    relearning_changes = []
    for index, task in enumerate(tasks):
        initial = mean(result["scores"][STAGES[index]][task]["accuracy"] for result in results.values())
        before = mean(result["scores"][endpoint][task]["accuracy"] for result in results.values())
        after = mean(result["scores"][relearn][task]["accuracy"] for result in results.values())
        relearning_changes.append([task, initial, before, after, 100 * (after - before), 100 * (after - initial)])
    table(sheet, "A 再学习对各域的影响（种子均值）", ["测试域", "本域初训后", "A 再学习前", "A 再学习后",
          "相对再学习前 (pp)", "相对本域初训 (pp)"], relearning_changes,
          {2: "0.00%", 3: "0.00%", 4: "0.00%", 5: "0.00", 6: "0.00"})

    budgets = []
    for seed, result in results.items():
        for stage in [*STAGES[:len(tasks)], relearn]:
            entry = result["training_budgets"][stage]
            task = entry["task"]
            budgets.append([seed, STAGE_LABELS.get(stage, "A 再学习"), task, entry["train_windows"],
                            entry["unique_windows_used"], entry["windows_seen"], entry["tail_dropped"],
                            entry["budget_skipped"], entry["update_steps"],
                            sum(manifests[task]["counts"]["validation"].values()),
                            sum(manifests[task]["counts"]["test"].values())])
    table(sheet, "各阶段实际训练量", ["种子", "阶段", "训练域", "可用训练窗口", "见过的不同窗口", "累计训练窗口",
          "丢弃尾批窗口", "预算未取用", "优化器步数", "验证窗口", "测试窗口"], budgets, {})
    for seed, rows in raw_rows.items():
        table(sheet, f"种子 {seed} 逐阶段测试结果", HEADERS, rows, score_formats)
    for index, width in enumerate((19, 18, 13, 18, 20, 17, 19, 20, 17, 22, 15, 13, 13), 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "C3"
    for tab in workbook:
        tab.sheet_view.showGridLines = False
    workbook.active = 1
    path = reports / f"{experiment}遗忘实验对比.xlsx"
    workbook.save(path)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=OUTPUT)
    parser.add_argument("--reports", type=Path, default=ROOT / "报告")
    args = parser.parse_args()
    results, config, manifests = [json.loads((args.run / filename).read_text(encoding="utf-8"))
                                  for filename in ("stage_metrics.json", "run_config.json", "split_manifests.json")]
    args.reports.mkdir(parents=True, exist_ok=True)
    for experiment in ("ABC", "ABCD"):
        export(experiment, results, config, manifests, args.reports)


if __name__ == "__main__":
    main()
