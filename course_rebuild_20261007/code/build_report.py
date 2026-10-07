"""Chinese research companion. Does not claim to be the final D3 submission."""
from pathlib import Path
import json,html
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/MovieLens_Course_Rebuild_CN.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('CJK','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
ink=HexColor('#243343');accent=HexColor('#b94e3d');teal=HexColor('#277b83');muted=HexColor('#667482')
W,H=A4;content=W-88
styles={
 'title':ParagraphStyle('Title',fontName='CJK',fontSize=24,leading=31,textColor=ink,spaceAfter=10),
 'h':ParagraphStyle('Heading',fontName='CJK',fontSize=16,leading=22,textColor=accent,spaceAfter=12),
 'sub':ParagraphStyle('Sub',fontName='CJK',fontSize=12,leading=17,textColor=teal,spaceBefore=8,spaceAfter=5),
 'body':ParagraphStyle('Body',fontName='CJK',fontSize=10.5,leading=16,textColor=ink,spaceAfter=8),
 'small':ParagraphStyle('Small',fontName='CJK',fontSize=8.5,leading=12,textColor=muted,spaceAfter=5),
 'cell':ParagraphStyle('Cell',fontName='CJK',fontSize=9,leading=13,textColor=ink),
 'headcell':ParagraphStyle('HeadCell',fontName='CJK',fontSize=9.5,leading=13,textColor=white),
}
for style in styles.values():style.wordWrap='CJK'
story=[]
def p(text,kind='body'):
    story.append(Paragraph(text,styles[kind]))
def fig(name,height):
    from PIL import Image as PILImage
    path=ROOT/'figures'/f'{name}.png';a=PILImage.open(path);width=min(content,height*a.width/a.height)
    story.append(Image(str(path),width=width,height=width*a.height/a.width));story.append(Spacer(1,7))
def table(headers,rows,widths=None):
    data=[[Paragraph(html.escape(str(x)),styles['headcell']) for x in headers]]+[[Paragraph(html.escape(str(x)),styles['cell']) for x in row] for row in rows]
    t=Table(data,colWidths=widths or [content/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#f4f6f7'),white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,ink)]))
    story.append(t);story.append(Spacer(1,10))
def newpage(title):story.append(PageBreak());p(title,'h')
def source(text):p('依据：'+text,'small')

p('MovieLens 课程复现<br/>与时间戳分析','title')
p('Decision 618 | 2026-10-07 | 研究与方法附件','small')
p('我们复现了老师的随机评分补全实验，并在相同评分目标上增加了时间切分和时间特征消融。完整 400 次新 CV 拟合已完成；新的 notebook 已执行，所有最终预测均按原始行号核对。')
p('老师的切分原则','sub')
p('Lecture 2 建议：有时间依赖时留出最近的数据；没有时间依赖时可以随机抽样。Lecture 7 的 MovieLens 使用 99/1 随机切分，适合课堂评分补全基准。两者对应不同的预测场景。')
table(['模型','随机测试 RMSE','未来测试 RMSE'],[
 ['训练均值','1.1237','1.0893'],['协同过滤 CF','0.9198','1.2246'],['CF + 静态信息','0.8990','1.0303'],['CF + 静态信息 + 时间','0.8940','1.0001']], [content*.44,content*.28,content*.28])
fig('model_comparison',183)
p('时间特征的独立增益应与同一测试集的静态集成比较：随机测试 RMSE 下降 0.0050，未来测试下降 0.0302。随机与时间方案的数据构成不同，不能用两列数值直接判断哪个实验更“准确”。')
source('Lec2_fall2026.pdf，PDF p.65；lec7_clusteringCF.pdf，PDF pp.47、52、55-59；results/model_metrics.csv。所有页码均为 PDF 位置，可能与页脚不同。')

newpage('1 / 各门课程如何切分数据')
p('检查范围：9 份技术讲义（1,154 页）、1 份嘉宾讲义、课程大纲、项目说明及 23 个 notebook。Canvas 当前 Class 11 / 12 模块没有新文件。完整文件 ID 与 SHA-256 在 sources/course_manifest.json。')
table(['课程 / 案例','老师的做法','项目中的应用'],[
 ['L1 网络案例','随机约 70/30；p.121','样本外评价'],
 ['L2 回归','p.65 给出时间 / 随机 / 分层原则；Solar 读取已切分 CSV','按业务时间依赖选择切分'],
 ['L3 流失分类','分层 70/30；seed=2025；p.29','保持类别比例'],
 ['L4 CART','医疗随机 70/30；Parole 分层 70/30；训练内 CV','调参后重训完整 training'],
 ['L5 RF / Boosting','70/30；RF 训练内 10 折；Boosting 代码快跑为 2 折','报告实际运行配置'],
 ['L6 特征 / 正则化','练习故意 25/75；正则化给定 70/30、10 折 CV','训练期拟合转换与 penalty'],
 ['L7 CF / Clustering','CF 随机 99/1、训练内 20 折；描述性 clustering 无监督式切分','保留课堂 CF，再加时间评价'],
 ['L8 神经网络','MNIST 官方 60k / 10k；CNN 演示复用 test 作 validation','项目保留独立最终 test'],
 ['L10 / 嘉宾','优化与业务应用；未新增预测切分规则','连接误差与业务使用方式']], [content*.23,content*.45,content*.32])
p('训练、验证、测试分别做什么','sub')
p('训练集估计模型参数；训练内部 CV 或单独 validation 选择模型；最终 test 评价选定方案。Scaler、编码和数值裁剪范围都在相应训练期确定。时间预测中的拟合数据必须早于被预测区间。')
p('99/1 和 25/75 都是特定示范设置。课件并没有要求项目统一使用某个比例。Lecture 4 p.135 的 “maximizes average loss” 是笔误，p.136 与代码的正确选择是最小化验证损失。')
source('各原始 PDF / notebook 的精确页码与 cell 索引见 Course_split_guide_CN.md。Notebook cell 索引从 0 开始。')

newpage('2 / 忠实复现课堂 MovieLens')
p('原数据：1,000,209 条评分、6,040 名用户、3,706 部被评分电影，目录有 3,883 部。评分范围 1-5；无缺失、重复 user-movie 对或未能关联的主键。')
p('原始随机切分为 990,206 / 10,003，NumPy seed=144。使用旧 RandomState，精确匹配原行号和 20 折编号。R softImpute 使用 lambda=0、maxit=1000，试 rank=1-20，预测截断至 1-5。缓存的全部 20 组指标经过独立复算，选 rank=8。')
fresh=pd.read_csv(ROOT/'results/fresh_selected_fit.csv').iloc[0]
table(['最终拟合','RMSE','MAE','OSR²'],[
 ['老师 notebook 已保存输出','0.919','0.709','0.332'],['本次课堂 rank=8 重训','0.919767','0.710589','0.330010'],['新 CV 选 rank=9 后重训',f'{fresh.RMSE:.6f}',f'{fresh.MAE:.6f}',f'{fresh.OSR2:.6f}']], [content*.46,content*.18,content*.18,content*.18])
fig('cv_curve',195)
p('新 CV 共 400 次 R 拟合，选 rank=9（CV RMSE=0.918274）；rank=8 为 0.918956，差 0.000682。新 CV 选择的 rank=9 在最终测试上并未超过课堂 rank=8；保留选择过程，不能根据 test 反复改 rank。')
p('原 notebook 未设 R seed 或记录包版本。本次使用 R 4.0.2 / softImpute 1.4-1；最终拟合 R seed=144，新 CV seed=144+100×fold+rank。因此流程复现成立，但不宣称与老师逐位相同。slide 的 0.916 也是另一轮运行。')
p('OSR² = 1 - sum((测试评分 - 预测)²) / sum((测试评分 - 训练均值)²)。分母沿用训练均值，不能替换为测试均值。')
source('ColFil_dist.ipynb，cells 5-15；L2 PDF pp.87-89；L7 PDF pp.45、47、50、52；fresh_cv_metrics.csv 与各 fit.csv。')

newpage('3 / 只使用当时能看到的数据')
p('随机测试中 98.85% 的评分，在同一用户的 training 中还有更晚的记录。它适合缺失评分补全；对于未来预测，需要另设严格的时间实验。切分只看 timestamp 的 80% / 90% 分位点，不看评分，并将同秒记录保留在同一侧。')
table(['阶段','评分数','UTC 区间'],[
 ['Train','800,164','2000-04-25 至 2000-12-02'],['Validation','100,024','2000-12-02 至 2000-12-29'],['Test','100,021','2000-12-29 至 2003-02-28']], [content*.22,content*.20,content*.58])
p('时间 CF 候选固定为 rank={4,8,12}、lambda={0,20}。Validation 选择 rank=4 / lambda=0；最后在前 900,188 条重训并评价 test。冷启动回退使用训练均值和 shrinkage=20 的用户 / 电影边际偏差，未知 ID 不删掉。')
p('第二层使用 forward 折外预测：前 50% 拟合并预测 50%-65%；前 65% 拟合并预测 65%-80%。80%-90% validation 调参后，也加入第二层校准数据。CF 对某条校准评分的预测没有用到该评分的训练标签。')
fig('warm_cold_population',180)
p('Validation 只有 23.84% warm，最终 test 则有 95.70%。用户构成、测试时长和评分分布变化很大，不能把两种切分的全部误差差异归因于未来信息。时间 CF 在 warm 为主的 test 上较差；直接因子点积与行号审计通过后，保留其负 OSR² 结果。')
source('temporal_split.json（精确到秒的边界）；temporal_validation_grid.csv；temporal_population.csv；independent_r_audit.csv。')

newpage('4 / 单独检验时间特征的价值')
p('老师 L7 已提出日期 / 时间集成。本次重建的是一个正则化线性 ensemble，沿用 L6 的 Ridge；老师 notebook 没有提供 slide ensemble 的完整代码，因此不声称还原老师的全部 51 个变量与参数。')
table(['版本','输入'],[
 ['静态集成','折外 CF 预测、18 类 genre、发行年份、用户性别 / 年龄类别 / 职业类别'],['时间集成','静态输入 + UTC 小时 / 星期、月份正余弦、经过天数、严格过去的训练期评分次数和最后评分间隔']], [content*.23,content*.77])
p('Ridge alpha 固定候选为 1、100、10,000。随机方案按用户分组作训练内部验证；其 CF 没有在该内部验证中再次嵌套拟合。最终 1% test 保持独立。时间方案按过去区间生成 OOF，最终 test 不参与特征拟合和调参。')
fig('timestamp_increment',185)
p('相对静态集成，时间特征在随机测试降低 RMSE 0.005022（0.56%），在未来测试降低 0.030223（2.93%）。按用户做 2,000 次配对 bootstrap，95% 区间分别为 [0.0031, 0.0069] 与 [0.0111, 0.0504]。区间以这次切分和模型选择为条件，不包括算法 / 调参流程的不确定性。')
p('预测时刻及静态元数据被假设为当时可用。计数与间隔只看已拟合时期中严格早于当前时刻的记录；同秒排除。数值 drift 特征按第二层训练范围裁剪，测试期不在线更新历史。UTC hour 不能解释为用户当地观影小时。')
p('结果支持离线评分误差下降，尚未检验 Top-N 排序、观看时长、留存或收入。下一步的业务验证需要统一的 ranking 实验或线上测试。')
source('L7 PDF pp.55-59；L6 正则化；evaluate_timestamp.py；timestamp_experiment.json。')

newpage('5 / 时间数据本身告诉了我们什么')
fig('monthly_time',225)
p('评分录入集中于 2000 年，后续月份样本小很多。月均评分同时受用户与电影构成影响，不能把它当成总体偏好变化的因果证据。全量数据用于这张描述性图；预测器仍只用其规定的训练期。')
fig('timestamp_ties',175)
p('994,169 个用户内相邻评分对中，529,046 对同秒（53.21%），88.72% 相隔不超过一分钟。按 30 分钟断点描述评分录入批次，共 25,163 个，中位数 6 条、90% 分位 115 条。该阈值是描述性选择，不能称为观影 session。')
p('如果继续 NextItNet，必须说明同秒记录无法确定顺序；以 MovieID 排序是人为规则。时间戳可以帮助建立历史 / 未来边界，但不提供真实观影序列，也不能把 ranking 指标与这里的星级 RMSE 直接比较。')
source('GroupLens README 的 timestamp 定义；timestamp_eda.json、rating_batches.json、monthly_activity.csv。')

newpage('6 / 解释、复跑与项目整合')
fig('genre_factors',205)
p('类别因子图沿用老师 cells 17-23 的 genre 均值、行内缩放和符号规则。各因子的方向与顺序不唯一，图中 [-5,5] 的值是缩放后的载荷，不是评分预测或固定客群标签。')
p('已经完成的审计','sub')
p('所有评分与预测按原始行号匹配；训练 / 验证 / 测试无重叠；时间边界严格有序；同秒不跨分区；独立重算全部 24 行指标；保存的集成模型在独立进程中可重现预测；完整 CF 预测与直接 UDV 点积一致；所有 400 个 CV 拟合的 fold / rank 唯一且无记录警告。')
p('如何接入 D3','sub')
p('推荐的叙述顺序是：Media & Entertainment 的候选电影评分问题，课堂 CF 复现，时间数据与评价场景，静态 / 时间特征的消融比较，部署所需的后续验证。可视化与预测评价两个任务均已完成，对应 L2、L4、L6 与 L7。')
p('本文件是研究与方法附件。旧 NextItNet 报告使用另一预测目标，不能直接替换其指标表。最终 D3 仍须自包含、最多 10 页、12pt、双倍行距，并另交数据、注释清楚的代码和 slides；贡献附录应记录成员真实完成的工作。')
p('文件入口','sub')
p('MovieLens_course_rebuild.ipynb / .html：已执行的英文 notebook 与中文解释。<br/>Course_split_guide_CN.md：完整课程切分依据。<br/>Methods_and_findings_EN.md：英文方法与结果。<br/>code/run_all.py：顺序复跑所有模型；--fresh-cv 再跑 400 次 CV。<br/>results/analysis_audit.json：数据、指标与特征审计记录。')
source('TermProjectInstructions.pdf，PDF pp.1-3；course_manifest.json。此目录保存在本地，未对外发布或提交到 Canvas。')

def footer(c,doc):
    c.saveState();c.setStrokeColor(HexColor('#e0e5e8'));c.line(44,38,W-44,38)
    c.setFont('Helvetica',8);c.setFillColor(muted);c.drawString(44,24,'DECISION 618 | MOVIELENS METHODS COMPANION | 2026-10-07');c.drawRightString(W-44,24,str(doc.page));c.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=43,bottomMargin=52,title='MovieLens Course Replication and Timestamp Analysis',author='Decision 618 project research',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
