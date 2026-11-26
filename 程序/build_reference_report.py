import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "程序/模型与日志/current_ABC_ABCD"
ANTI = ROOT / "程序/模型与日志/anti_forgetting/stage_metrics.json"
OUT = ROOT / "报告/四数据集遗忘现象实验参考.docx"
FIG = ROOT / "报告/figures"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def avg_record(records):
    return {
        "accuracy": mean(r["accuracy"] for r in records),
        "macro_recall": mean(r["macro_recall"] for r in records),
        "loss": mean(r["loss"] for r in records),
        "normal": mean(r["per_class_recall"]["normal"] for r in records),
        "inner": mean(r["per_class_recall"]["inner_race"] for r in records),
        "outer": mean(r["per_class_recall"]["outer_race"] for r in records),
    }


def load_data():
    baseline = json.loads((RUN / "stage_metrics.json").read_text(encoding="utf-8"))
    anti = json.loads(ANTI.read_text(encoding="utf-8"))
    stages = ["initial_A", "after_B", "after_C", "after_D"]
    tasks = ["A_CWRU", "B_JNU", "C_UESTC", "D_Ottawa"]
    baseline_avg = {
        stage: {task: avg_record([v["scores"][stage][task] for v in baseline.values()])
                for task in tasks}
        for stage in stages
    }
    return baseline, anti, baseline_avg, stages, tasks


def plot_figures(baseline_avg, anti):
    FIG.mkdir(parents=True, exist_ok=True)
    labels = ["A初训", "B后", "C后", "D后"]
    stages = ["initial_A", "after_B", "after_C", "after_D"]
    domains = ["A_CWRU", "B_JNU", "C_UESTC", "D_Ottawa"]
    names = ["A-CWRU", "B-JNU", "C-UESTC", "D-Ottawa"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    for domain, name in zip(domains, names):
        ax.plot(labels, [100 * baseline_avg[s][domain]["accuracy"] for s in stages], marker="o", linewidth=2, label=name)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("训练阶段")
    ax.set_title("固定测试集上的跨域顺序性能轨迹")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    trajectory = FIG / "跨域顺序Accuracy轨迹.png"
    fig.savefig(trajectory, bbox_inches="tight")
    plt.close(fig)

    methods = ["sequential", "ewc", "pq_replay", "uciln"]
    method_names = ["顺序微调", "EWC", "PQ回放", "UCILN"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    x = np.arange(len(methods))
    width = .19
    for i, (domain, name) in enumerate(zip(domains, names)):
        values = [100 * mean(v["scores"]["after_D"][domain]["accuracy"] for v in anti[m].values()) for m in methods]
        ax.bar(x + (i - 1.5) * width, values, width, label=name)
    ax.set_xticks(x, method_names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("after D Accuracy (%)")
    ax.set_title("四域末端的抗遗忘方法对比")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    endpoint = FIG / "抗遗忘方法四域末端对比.png"
    fig.savefig(endpoint, bbox_inches="tight")
    plt.close(fig)
    return trajectory, endpoint


def set_cell(cell, text, bold=False):
    cell.text = str(text)
    for run in cell.paragraphs[0].runs:
        run.font.name = "Noto Sans CJK JP"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK JP")
        run.font.size = Pt(9)
        run.bold = bold


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for cell, text in zip(t.rows[0].cells, headers):
        set_cell(cell, text, True)
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            set_cell(cell, text)
    return t


def pct(x):
    return f"{100*x:.2f}%"


def pp(x):
    return f"{x:.2f}"


def build_doc(baseline, anti, avg, stages, tasks, trajectory, endpoint):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.65)
    sec.bottom_margin = Inches(.65)
    styles = doc.styles
    styles["Normal"].font.name = "Noto Sans CJK JP"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK JP")
    styles["Normal"].font.size = Pt(10.5)
    title = doc.add_heading("跨数据集顺序训练中的遗忘现象实验参考文档", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("实验范围：A=CWRU → B=JNU → C=UESTC → D=Ottawa v2；主问题是跨域适配后的旧域性能下降。本文档对应当前最新状态，旧用户原稿“持续学习(遗忘探索).docx”保持不变。")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("1. 实验目的", 1)
    doc.add_paragraph("本实验探索同一个三分类故障诊断模型在多个数据集顺序适配后是否发生遗忘，以及遗忘在哪个域、哪一类标签和哪个训练阶段出现。实验先建立跨域顺序微调基线，再作为后续抗遗忘方法比较的参照；当前结论不把它扩展为已经解决持续学习问题。")
    doc.add_paragraph("实验对象是固定标签域增量：各数据集均使用 normal、inner-race、outer-race 三类，改变的是采集设备、采样率、转速、信号分布和背景工况。因此更准确的定位是“跨数据集、固定标签域增量下的顺序微调遗忘基线”，而不是新增故障类别的 class-incremental benchmark。")

    doc.add_heading("2. 实验设置", 1)
    doc.add_heading("2.1 设备与软件", 2)
    doc.add_paragraph("训练环境记录于程序日志：Intel Core i7-14650HX，Ubuntu 26.04.1 LTS，Python 3.12.0，PyTorch 2.14.0.dev20260803+cu130。实验记录同时保留 GPU 基线和 CPU 对照，CPU 线程数为 2。")
    doc.add_heading("2.2 数据集对比", 2)
    table(doc, ["域", "数据集与背景", "原始条件", "本实验处理"], [
        ["A", "CWRU 轴承数据集", "12 kHz drive-end，0 HP", "三类；按原始记录时间 70/15/15 划分"],
        ["B", "JNU Bearing Dataset", "600 rpm，50 kHz", "三类；重采样到 12 kHz"],
        ["C", "UESTC Bearing Dataset", "800 rpm，20 kHz", "三类；重采样到 12 kHz"],
        ["D", "Ottawa Bearing Dataset v2", "200 kHz；四种变速曲线，三次试验", "健康/内圈/外圈；按完整 trial 1/2/3 划分"],
    ])
    doc.add_paragraph("统一输入为 2048 点窗口、步长 512、逐窗口 z-score，标签映射为 normal=0、inner-race=1、outer-race=2。A/B/C 的测试窗口数为 132、166、129，D 的测试窗口数为 2772。A/B/C 来自所选原始记录的时段划分，不能把相邻窗口视为独立轴承重复；D 的 trial 留出更接近独立记录测试。")
    doc.add_heading("2.3 训练与测量协议", 2)
    doc.add_paragraph("模型为 WDCNN 三分类器。共享状态包括卷积层、BatchNorm 仿射参数和三分类头；输入采样率、窗口、标准化、标签映射及 A 域 BatchNorm 运行统计属于可切换输入/评估配置。A 训练后固定初始检查点，按 A→B→C→D 顺序每域 100 个优化器更新步，Adam 学习率 0.001、批大小 64，种子 2026/2027/2028。每个阶段都在 A、B、C、D 固定测试集评估。")
    doc.add_paragraph("主测量是无更新测量：切换声明的域输入配置，关闭梯度、优化器更新和 BatchNorm 运行统计更新，不重新加载初始 A 权重。A 再训练分支另行保留，用于说明重新学习可以恢复性能，但不能当作无更新配置切回的结果。")

    doc.add_heading("3. 实验结果", 1)
    doc.add_paragraph("以下两表为顺序微调基线的种子均值；完整指标、逐种子结果和抗遗忘方法表在同目录 XLSX 文件中。Accuracy、Macro Recall、Loss 和三类 Recall 同时保留，遗忘量按旧域历史最佳性能计算。")
    for exp, domain_count, title_text in [("ABC", 3, "ABC 实验：A→B→C"), ("ABCD", 4, "ABCD 实验：A→B→C→D")]:
        doc.add_heading(title_text, 2)
        selected = tasks[:domain_count]
        rows = []
        for si, stage in enumerate(stages[:domain_count]):
            for task in selected:
                r = avg[stage][task]
                rows.append([stage.replace("initial_A", "初训 A").replace("after_B", "B 后").replace("after_C", "C 后").replace("after_D", "D 后"), task, pct(r["accuracy"]), pct(r["macro_recall"]), f"{r['loss']:.4f}", pct(r["normal"]), pct(r["inner"]), pct(r["outer"])])
            if si:
                old = selected[:si]
                d = []
                f = []
                for task in old:
                    first_stage = stages[selected.index(task)]
                    for result in baseline.values():
                        current = result["scores"][stage][task]["accuracy"]
                        first = result["scores"][first_stage][task]["accuracy"]
                        prior = [result["scores"][s][task]["accuracy"]
                                 for s in stages[selected.index(task):si]]
                        d.append(100 * (first - current))
                        f.append(100 * (max(prior) - current))
                rows.append([stage.replace("after_B", "B 后").replace("after_C", "C 后").replace("after_D", "D 后"), "旧域汇总", "—", "—", "—", "—", "—", f"下降={mean(d):.2f} pp；AF={mean(f):.2f} pp；BWT={-mean(d):.2f} pp"])
        table(doc, ["阶段", "测试域", "Accuracy", "Macro Recall", "Loss", "正常 Recall", "内圈 Recall", "外圈 Recall"], rows)
    doc.add_picture(str(trajectory), width=Inches(6.75))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("图 1. 固定测试集上的 Accuracy 轨迹。")

    doc.add_heading("4. 实验结论", 1)
    a_abc = 100 * avg["after_C"]["A_CWRU"]["accuracy"]
    a_abcd = 100 * avg["after_D"]["A_CWRU"]["accuracy"]
    doc.add_paragraph(f"A 域初训 Accuracy 为 100.00%；经过 B、C 后，ABC 末端 A 为 {a_abc:.2f}%，相对初训下降 {100-a_abc:.2f} 个百分点；经过 D 后，ABCD 末端 A 为 {a_abcd:.2f}%，下降 {100-a_abcd:.2f} 个百分点。下降同时体现为 Macro Recall 和逐类 Recall 的变化，说明不是单一总体比例波动。")
    doc.add_paragraph("遗忘具有阶段性和类别不均衡性：A 的不同故障类别 Recall 下降幅度不同，表明跨域更新改变了决策边界和特征表示，而不是所有类别等比例损失。B、C 在后续阶段也可能下降；D 的当前域性能高并不代表模型没有遗忘，只说明模型对最近适配域仍具有可塑性。")
    doc.add_paragraph("原因目前只能做机制层面的解释：跨数据集信号分布、采样率和工况差异导致梯度方向冲突；共享卷积表示和分类头被后续域更新；BatchNorm 运行统计切换会改变输入特征分布。现有实验不能仅凭性能表证明某一个原因是唯一原因，因此报告把这些作为待验证机制，而不是定论。")

    doc.add_heading("5. 解决方法", 1)
    doc.add_heading("5.1 UCILN 方法复现与四域适配", 2)
    doc.add_paragraph("本项目参考 Zhan 等人的 UCILN 论文：以源域有标签、目标域无标签的跨域类增量学习为主线，使用特征空间约束、PQ 记忆/回放和多核 MMD 近似域对齐。当前代码是独立实现和协议适配，不宣称复现论文原始 benchmark 数值：论文的类增量设置与本实验固定三类的四数据集顺序域适配并不相同。")
    doc.add_paragraph("四域实现中保留 A 源域记忆，并对先前域做有限特征记忆；采用 PQ 近似记忆、交叉熵与高斯多核 MMD，冻结部分特征/BN 并更新分类相关参数。结果应看作抗遗忘探索对照，而不是严格公平排名，因为 UCILN、EWC 和顺序微调对目标域标签的使用方式不同。")
    doc.add_heading("5.2 EWC 方法复现", 2)
    doc.add_paragraph("EWC 在每个旧域保存参数快照和经验 Fisher 对角近似，通过惩罚重要参数偏离旧任务参数来保持旧知识。本实验使用每域 128 个训练窗口估计 Fisher，λ=1000；不保存旧波形，仅保存参数与 Fisher。它是监督式当前域适配基线，和无标签目标域的 UCILN 不具备完全相同的监督预算。")
    doc.add_heading("5.3 当前方法结果与解释", 2)
    method_rows = []
    for method, name in [("sequential", "顺序微调"), ("ewc", "EWC"), ("pq_replay", "PQ 特征回放"), ("uciln", "UCILN")]:
        vals = [100 * mean(v["scores"]["after_D"][d]["accuracy"] for v in anti[method].values()) for d in tasks]
        method_rows.append([name, *[f"{v:.2f}%" for v in vals], f"{vals[0]-100:.2f} pp"])
    table(doc, ["方法", "A末端", "B末端", "C末端", "D末端", "A相对初训"], method_rows)
    doc.add_picture(str(endpoint), width=Inches(6.75))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("图 2. 抗遗忘对照的 D 后固定测试集结果。")
    doc.add_paragraph("本次结果显示，UCILN 在当前实现中能明显保护 A，但 D 的新域性能接近三分类随机水平；PQ 回放能保护 A，却没有学习当前域；EWC 在部分域保留上有改善，但不是所有域都改善。因此当前阶段不应宣布任何方法已经解决遗忘，而应继续做监督预算、记忆预算、BN 状态和更新步数的严格对照。持续学习可简述为：模型在任务/域序列上持续获得新数据，同时尽量保持旧任务性能，核心矛盾是稳定性与可塑性。")
    doc.add_paragraph("直接复现/参考文献：Zhan et al., “Mitigating Catastrophic Forgetting in Cross-Domain Fault Diagnosis: An Unsupervised Class Incremental Learning Network Approach,” IEEE Transactions on Instrumentation and Measurement, 2025, DOI: 10.1109/TIM.2024.3500047；Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” PNAS, 2017, DOI: 10.1073/pnas.1611835114（EWC）。近三年延伸阅读清单见“文献/近三年遗忘分析论文阅读清单.md”。")

    doc.add_heading("6. 问题讨论", 1)
    for text in [
        "数据独立性：A/B/C 使用单条所选记录的时间切分，窗口高度相关，结论限定为记录内时段泛化；后续应按独立轴承、独立试验或原始记录分组留出。",
        "训练预算：各域按相同步数而非相同样本数，D 的样本量明显大于 A/B/C；当前应标注为等更新步数、非等样本量。后续需要报告实际样本数、丢弃样本数、更新步数，并增加等样本或等更新预算对照。",
        "配置与权重：当前 no-update 测量没有重新加载 A 权重，避免把回滚检查点误称为遗忘测量；A 再训练是独立实验臂。仍需进一步把共享 BN、域专属 BN 和真正可训练 adapter 分开做消融。",
        "文献复现边界：UCILN 原始论文是无监督类增量协议，本项目将其改造成四域固定三类探索版本，必须在报告中明确“方法思想复现/协议适配”，不能声称原论文 benchmark 的完全复现。",
    ]:
        doc.add_paragraph(text, style="List Bullet")
    doc.add_paragraph("图示提示词（如需交给 GPT Image）：绘制一张论文风格的横向流程图，左侧是 CWRU A 的三类振动波形和 WDCNN，共享模型位于中央；右侧依次连接 JNU B、UESTC C、Ottawa D，底部用红色但克制的折线表示旧域 Accuracy 下降，另用灰色分支表示切回 A 配置但不更新权重；不生成数值，不使用装饰性渐变，不替代实测图表。")
    doc.add_paragraph("主要文件：训练脚本与模型在“程序/”；数据与配置在“数据集/”；本参考文档、Excel 和图表在“报告/”；直接使用的原文及近三年阅读清单在“文献/”。")
    doc.save(OUT)


if __name__ == "__main__":
    baseline, anti, avg, stages, tasks = load_data()
    trajectory, endpoint = plot_figures(avg, anti)
    build_doc(baseline, anti, avg, stages, tasks, trajectory, endpoint)
    print(OUT)
