本文件夹内容说明（整理时间：2026-10-11）
================================================

这是从原文件夹"硫苷所有的数据和代码"里，挑出这篇NIR硫苷分类稿件(Broccoli
glucosinolate NIR manuscript)实际用到的数据和代码，单独打包出来的。原文件夹
体量大、历史版本多，这份是精简后可独立使用的子集。

reproducibility_v2/  —— 核心可复现分析包（即manuscript/Broccoli_NIR_reproducibility_v2.zip解压后的内容）
  README.md                                本包自带的详细说明（数据范围、运行入口、依赖、解读注意事项）
  manifest.json                            文件清单
  74samples.csv / analysis_band/sample_labels.csv   74样本cohort的标签与来源
  analysis_band/head_confirmed_20261007/   确认过的head-only光谱(126份)、审计记录、
                                            23路算法在74样本上的完整结果
  analysis_band/*.py                       74样本分析相关脚本（分类、分段、算法对比等）
  manuscript/sensitivity_88.py             88样本敏感性分析的运行入口（本次v3修订新增内容的数据来源）
  manuscript/sensitivity_88/                上面脚本跑出的结果：labels.csv（88样本真实浓度和分组，
                                            已经核实过）、summary.csv（各模型/区间的平衡准确率等指标）、
                                            fold_metrics.csv、oof_predictions.csv等
  manuscript/verify_sensitivity.py          一致性回归检查脚本（确认重跑结果与保存结果一致，最大误差<6e-11）
  历史/126种青花菜硫苷成分.xlsx              corrected化学数据工作表

raw_inputs/  —— 最原始的两份输入文件（reproducibility_v2里的head_spectra_126.csv和
                 sample_labels.csv就是从这两份文件处理出来的）
  TQData.csv          近红外原始光谱导出文件（含head和leaf两种组织，GBK编码）
  glucosinolate.xls   硫苷浓度原始化学数据表

manuscripts/  —— 稿件三个版本
  Broccoli_glucosinolate_NIR_manuscript_EN_v1.docx   初稿
  Broccoli_glucosinolate_NIR_manuscript_EN_v2.docx   补充88样本敏感性分析、加引用后的修订稿
  Broccoli_glucosinolate_NIR_manuscript_EN_v3.docx   当前最新稿：Table 1补了88样本cohort统计
                                                      （真实计算值，非编造），新增Figure 4
                                                      （74 vs 88样本对比图），修了连字符

reference/  —— 稿件里引用、对比的参考文献
  Yijun_zhang_glucosinolate_2025_FoodChemX.pdf   Zhang et al. 2025, Food Chemistry: X
                                                  （稿件正文里反复引用的"reference study"）

【重要提醒，来自reproducibility_v2/README.md】
- 这是本地审阅用的复现包，不等于已授权对外公开发布的数据仓库
- 请在投稿前确认HPLC完整方法学、样品来源信息、作者声明、数据共享权限后，
  再决定是否/如何公开发布
- 部分历史脚本里保留了旧版"头叶混合"的数据加载逻辑，仅作溯源用，不要再运行；
  新的入口点(sensitivity_88.py、rerun_head_suite_20261007.py)已经是只用head-only数据
