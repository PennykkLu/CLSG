### CLSG：基于反事实用户行为生成的会话推荐方法

### 摘要
为保护用户隐私, 许多平台为用户提供了匿名登录选项, 迫使推荐系统仅能访问当前会话中的有限用户行为记录, 进而催生了会话推荐(Session-Based Recommendation, SBR)系统. 现有SBR技术在很大程度上沿用了传统非匿名用户行为建模思路, 聚焦于序列建模以习得会话表征. 然而, 当会话长度偏短时, 现有SBR技术性能衰减严重, 难以应对以短会话为主的真实会话推荐场景. 有鉴于此, 提出一种通过频繁模式引导长会话生成的反事实推理方法(**C**ounterfactual Inference by Frequent Pattern Guided **L**ong **S**ession **G**eneration, CLSG), 试图回答反事实问题: "如果会话内包含更丰富的交互物品, SBR模型预测结果将会如何?" CLSG遵循反事实理论的"归纳-行动-预测"经典三阶段推理流程. "归纳": 从已观测会话集合中构建频繁模式知识库; "行动": 基于所构建知识库生成反事实长会话; "预测": 度量已观测会话和反事实会话预测结果间的差异, 并将其作为正则化项并入目标函数, 以达到表征一致性的目的. 值得注意的是, CLSG具有模型无关的技术特点, 可对既有SBR模型实现普惠式赋能. 三个基准数据集上的实验结果表明, CLSG提升了五款既有SBR模型的预测性能, 在命中率(HR)和平均倒数排名(MRR)评价指标上均取得6%左右的平均性能提升.


### 依赖
```angular2html
python==3.7
pytorch==1.13.1
pytorch_lightning==1.2.6
cudatoolkit==10.1
```

### 数据集 DATASET 
```
- Diginetica
- Nowplaying
- Tmall
```

### 基模型 MODEL
```angular2html
- NARM [Li et al., Neural attentive session-based recommendation, CIKM, 2017]
- STAMP [Liu et al., STAMP: short-term attention/memory priority model for session-based recommendation, KDD, 2018]
- SRGNN [Wu et al., Session-based recommendation with graph neural networks, AAAI, 2019]
- CORE [Hou et al., CORE: simple and effective session-based recommendation within consistent representation space, SIGIR, 2022]
- AttMix [Zhang et al., Efficiently leveraging multi-level user intent for session-based recommendation via atten-mixer network, WSDM, 2023]
```

### 复现
```angular2html
python main.py --dataset DATASET --model MODEL
```
由于投稿系统文件上传限制，本代码仓库仅提供原始代码、Tmall数据集频繁模式及测试集文件，无法提供预先训练好的模型，更多细节将在论文被正式接收后进行完全公开。

