# DA-Net接入说明

**常规读者无需手工执行下文接入步骤。** prepare.py/train.py/evaluate.py已提供
自动接入入口，优先按READER_GUIDE_zh.md运行。下文仅供自定义基础工程时参考。

## 1. 基础实现单独获取

从 https://github.com/Rphone/DA-Net 取得代码，保存在本发布目录之外。
不要把基础源码复制进本仓库。记录所用版本：

```bash
python scripts/inspect_upstream.py --upstream /path/to/DA-Net
```

该脚本不会执行、复制或下载基础源码，只记录git提交（若存在）、模型SHA256及
必要接口的静态名称。它不是端到端兼容性测试。上游版本不同需要相应适配。

## 2. 接口合同

桥接器需要基础模型暴露下列模块：

| 接口 | 所需行为 |
|---|---|
| pc_linear | 每次基础前向调用一次，输出[B×K,1024] |
| pic_linear | 每次基础前向调用一次，输出[B×K,1024] |
| pic_backbone | 共享视觉编码器，输入[B,3,224,224]，输出[B,768] |
| fc | 评分头，每次前向调用一次，输入聚合后的[B,2048] |

B是点云批量，K是实际patch数。局部输入一般为pcs=[B,K,8192,6]、
pics=[B,K,3,224,224]。基础模型保留原始编码、拼接、patch聚合、评分头、
FDM/FMM、原有损失。本包没有复制这些实现。
仅支持单设备eager模式；未验证DDP、DataParallel、torch.compile或重入/梯度检查点。

## 3. 安装新增模块

初始化顺序为：设置随机种子、构建基础数据集/加载器、构建基础模型。
使用普通基础模型，不要启用已有GTFS/LGQC或其他实验分支。

```python
from dfgc.integration import DFGCExtensions

# host由使用者自行获得的基础项目构造，此处不分发其构造源码。
extension = DFGCExtensions(
    host, geometry_weight=0.02, texture_weight=0.02,
    beta=0.1, hidden_dim=256, preserve_cuda_rng=(dataset == "sjtu"),
)
# 接着host.to(device)，创建optimizer或载入同构联合checkpoint。
# optimizer必须在安装之后创建，并覆盖host.parameters()。
```

新增头和适配器直接注册在host上，随其迁移设备、切换train/eval及保存state_dict。
不要重复安装。建议CPU初始化时接入。因素头独立初始化种子2001，适配器3501；
全局前向使用独立RNG流。相同种子仍不保证不同数据读取器/硬件具有完全相同轨迹。

## 4. 数据与标签

保持基础数据集对patch、PNG、增强、MOS缩放和归一化的处理。
新增包装器只添加因素标签和拼图，不包含或替换原局部读取器。

```python
from dfgc.dataset_extension import ContextDataset

dataset = ContextDataset(base_dataset, names, factors, cache_root,
                         training=True, seed=42)
```

names必须与base_dataset行顺序完全一致，元素为不带.ply的样本名；不能另行排序。
base_dataset返回(name或metadata字典, pcs, pics, scaled_mos)。基础版本的返回格式
不同时，由使用者在自己的工程中适配。factors为{name: 因素标签字典}。

SJTU：

```python
from dfgc.sjtu_factors import factor_metadata
factors = {name: factor_metadata(name) for name in names}
```

WPC从使用者自己的元信息表读取生成参数：

```python
from dfgc.wpc_factors import factor_metadata
factors = {
    str(row.Ply_name).removesuffix(".ply"): factor_metadata(
        str(row.Distortion_Type), row.G_Parameter, row.T_Parameter)
    for row in rows
}
```

每个样本需geometry_severity、geometry_severity_mask、texture_severity、
texture_severity_mask四个字段。标签在[0,1]，mask为0或1；mask=0表示该因素
不参与监督，不是将其作为严重度0训练。标签来自生成配置，不根据MOS拟合。
SJTU使用六级相对严重度映射；WPC使用独立参数表。

## 5. 训练与验证调用位置

可执行函数示例见examples/integration_step.py。训练：

```python
metadata, pcs, pics, mos = batch  # 调用方负责移动到正确设备
with extension.batch(metadata["global_mosaic"], pcs.shape[0], pcs.shape[1],
                     epoch=current_epoch, step=global_step) as context:
    prediction, base_losses = host(pcs, pics, mos)
    factor_loss, terms = context.auxiliary_loss(metadata)
    loss = base_losses["loss"] + factor_loss
# 按基础训练器既定AMP/optimizer流程处理loss，不要重复backward。
```

示例假设基础forward返回(prediction,loss字典)。若版本不同，只调整提取其原始
总损失的调用点，不替换原MOS、VQ、重建及距离相关性损失，也不要重复加GTFS。

验证/推理同样启用LGQC，但不向验证总损失加入GTFS：

```python
import torch
host.eval()
with torch.no_grad():
    with extension.batch(metadata["global_mosaic"], pcs.shape[0], pcs.shape[1],
                         epoch=current_epoch, step=global_step):
        prediction, base_losses = host(pcs, pics, mos)
```

桥接器在fc输入处加入共享Swin生成的残差，并捕获1024维模态特征；没有复制
基础forward。临时hook会在正常结束和异常时移除。必须在with内部、基础前向
之后计算因素loss。不得detach全局特征或复制第二个Swin，不得在推理时遗漏上下文。
每个上下文只允许一次完整基础前向，维度/调用次数不符会报错。

使用Lightning时，在自己的training_step/validation_step中接入上述逻辑；
保留基础损失和优化器语义。本包的runtime.py及train.py已提供对应运行编排。

## 6. 配置、选模与权重

configs/*.json由本项目统一入口自动加载，不是上游main.py直接可读的通用配置。
自定义基础工程时才需手工映射路径、归一化、MOS缩放和训练超参数。
configs/splits/*.json保留逐折样本名与行顺序，用build_splits.py与用户MOS表合并。

参考设置：seed42、100轮、batch4、训练8patch/验证32patch；AdamW学习率5e-5、
weight_decay=1e-5，StepLR每5轮乘0.9；WPC MOS/100且不归一化XYZRGB，
SJTU MOS/10且按历史基础读取器归一化XYZRGB。扩展不实现局部归一化。
每轮评价留出折，按留出折最高SROCC选模；没有新增内部验证集。
保存top3及last、optimizer/scheduler/AMP状态、配置、上游版本与逐样本预测；
WPC五折/SJTU九折分别算术平均。统一训练入口已实现这些保存和续训步骤。

恢复联合权重前，先构建同构host并安装extension，再strict加载state_dict。
新增模块键名为gtfs_geometry_head、gtfs_texture_head、global_context_adapter。
仅载入基础权重时新增模块还未训练，不是最终模型。任意上游版本不一定能strict
加载历史完整权重，仍须检查所有基础参数名与形状。

## 7. 复现边界

历史WPC使用上游PNG缓存，未获得生成它们的原T-Net权重。新训T-Net能生成PNG，
但不能保证逐像素或最终指标相同。历史SJTU的T-Net使用全部无MOS几何数据，
包含质量留出内容；不得将该设置描述为严格按内容隔离的T-Net训练。

接入逻辑已通过CPU续训检查，以及WPC/SJTU第1折完整骨干GPU各4批训练/验证
冒烟测试；完整GPU交叉验证尚未重跑。使用不同基础版本时，应先运行完整性
检查及默认4批的冒烟测试，确认优化器实际更新。
