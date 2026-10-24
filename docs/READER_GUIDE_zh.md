# 读者复现指南：下载、放置、运行

[English](READER_GUIDE.md) | **简体中文**

本指南对应WPC和SJTU-PCQA上的最终DFGC-Net（GTFS＋LGQC）。
读者无需编辑DA-Net源码。我们的三个入口会从--upstream指定的位置导入基础组件。
所有命令均在DFGC-Net根目录运行；路径可按自己的电脑修改。

## 1. 下载哪些文件

| 下载项 | 获取入口 | 放置位置 |
|---|---|---|
| DFGC-Net代码 | 本仓库 | workspace/DFGC-Net/ |
| DA-Net基础代码 | [官方仓库](https://github.com/Rphone/DA-Net)，Code→Download ZIP，或git clone | workspace/DA-Net/，其中直接包含main.py、model/、data/ |
| Swin-T ImageNet-1K预训练权重 | [微软模型列表](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md)，第一行Swin-T、224×224、1K model | DFGC-Net/pretrained/swin_tiny_patch4_window7_224.pth |
| WPC原始失真点云 | [WPC官方入口](https://github.com/qdushl/Waterloo-Point-Cloud-Database)，README的Google Drive链接 | workspace/data/wpc/raw/，可包含子目录 |
| WPC历史32-patch及局部PNG缓存 | [DA-Net README](https://github.com/Rphone/DA-Net#download-the-preprocessed-wpc-datasets-and-pretrained-checkpoints)中的网盘链接 | PLY放data/wpc/patches/，PNG放data/wpc/projections/ |
| SJTU-PCQA数据与MOS | [数据集作者页面](https://vision.nju.edu.cn/28/fd/c29466a469245/page.htm)的下载入口 | 失真PLY放data/sjtu/distortion/；Final_MOS.mat放data/sjtu/ |

Swin官方直链：
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

DA-Net作者README所列WPC缓存入口（文件属于第三方，链接可能变化，以其README为准）：

- [百度网盘](https://pan.baidu.com/s/1qXzm8nF_H5ABftAJzsD1sw?pwd=wwd3)，提取码wwd3。
- [OneDrive](https://1drv.ms/f/c/6900cdc80191473b/IgBpMxxEKdAjT4V1KnnltpHQAZpYLgHz0ck6ZBPY04-dmPQ)。

SJTU作者页面所列入口：

- [百度网盘](https://pan.baidu.com/s/1x6ZqETv4b3O0a6LB0oV7jg)，提取码rzgk。
- [Google Drive](https://drive.google.com/file/d/13L1opI6dM5eiNkwy9pu0URfI09hyIV8a/view?usp=sharing)。

只需对应实验的失真样本：WPC740个，SJTU378个。不要把参考点云混入失真目录。
SJTU保留官方Final_MOS.mat；不要自行把没有列名的MOS.xlsx当作命名样本表读取。
WPC元信息由读者下载的DA-Net fold1两张表提取，再生成本项目提供的固定五折划分。
这些MOS表不打包在本仓库中；也可以通过--metadata提供合法获取的规范化元信息表。
无需下载DA-Net质量模型权重即可从头训练，只需Swin的上述预训练权重。

## 2. 目录示例

```text
workspace/
├── DFGC-Net/
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── pretrained/swin_tiny_patch4_window7_224.pth
├── DA-Net/
│   ├── main.py
│   ├── model/IQA_Model.py
│   ├── data/Datasets.py
│   └── configs/wpc/1/{train.xlsx,vail.xlsx}
└── data/
    ├── wpc/
    │   ├── raw/                    # 原始失真点云
    │   ├── patches/                # 32-patch缓存，可下载后放入
    │   └── projections/            # 对应局部PNG，可下载后放入
    └── sjtu/
        ├── distortion/             # 原始失真点云
        └── Final_MOS.mat
```

ZIP常会多解压一层目录。--upstream要指向直接包含main.py的那一层，而不是外层空壳。
不要把第三方基础代码放进DFGC-Net内部；也不要上传自己的data/、outputs/或权重。

## 3. 安装环境

推荐独立环境，参考配置为Python3.9、PyTorch2.2.0、CUDA11.8：

```bash
conda create -n dfgc python=3.9 -y
conda activate dfgc
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-reader.txt
```

另外安装与Python/PyTorch/CUDA匹配的PyTorch3D，见
[官方安装说明](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)。
Windows可使用与上述版本完全匹配的预编译wheel；本仓库不提供第三方wheel。
不能仅通过pip install一个版本未知的wheel来保证兼容。
基础DA-Net本身可能顶层导入PyTorch3D，因此使用现成PNG也应确认它能导入。

检查环境：

```bash
python -c "import torch; from pytorch3d import ops; print(torch.__version__, torch.cuda.is_available())"
python scripts/inspect_upstream.py --upstream ../DA-Net
python -B -m unittest discover -s tests -v
```

若出现“已安装扩展/名称冲突”，检查下载的是否为干净的基础DA-Net，而不是已经加入
其他GTFS/LGQC实验的修改版。基础代码缺少对应接口时不要忽略报错，应先核对版本。

## 4. WPC：使用论文对应的历史局部缓存

将下载的PLY/PNG缓存按第2节放好：不要把PC/proj外层目录再嵌套进去。
应直接存在patches/<name>_part0.ply和projections/<name>_part0_1.png，直到part31。
原始WPC失真PLY仍需下载，用于生成我们的全局六视图。

依次运行：

```bash
python prepare.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --stage metadata
python prepare.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --stage views
python prepare.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --stage check
python train.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --folds 1 --dry-run
python train.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --folds 1 --smoke
python train.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc
```

最后一条自动顺序训练1—5折。dry-run只检查输入；smoke默认训练/验证各4个批次，
并检查是否实际发生优化器更新，可用--smoke-batches调整。不用于报告性能，
也不会生成正式模型结果。AMP可能在初始阶段跳过更新来调整损失缩放，
因此仅跑一个批次不能可靠验证优化器更新。

## 5. SJTU：从原始点云完整生成输入再训练

在data/sjtu/放好distortion/和Final_MOS.mat后，依次运行：

```bash
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage metadata
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage patches
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage views
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage tnet
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage png
python prepare.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --stage check
python train.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu --folds 1 --smoke
python train.py --upstream ../DA-Net --dataset sjtu --data-root ../data/sjtu
```

最后一条自动顺序训练1—9折。各阶段输出均在对应data/sjtu/内部：
metadata.csv、splits/、patches/、global_views/、tnet_normals_k50/、tnet/和projections/。
patch阶段是按样本固定种子的串行调用，不依赖多进程调度；可能耗时较长。

T-Net阶段只用几何/法向量，不用MOS；每个点云28个patch用于训练，4个用于固定监控。
预处理包含质量评价留出内容的无标签几何数据。
最多100轮，监控损失10轮无明显改善早停。输出tnet/best.ckpt及last.ckpt；
png阶段自动加载best.ckpt，也可用--tnet-checkpoint指定另一个合法获得的对应权重。
T-Net中断后用相同命令继续，不能为了“续训”随意改变配方参数。
新生成缓存带记录；程序拒绝覆盖来源不明的旧缓存或混用新旧配方。

## 6. WPC也可以从原始数据生成

在一个没有导入局部缓存的新WPC数据根目录，按第5节顺序运行全部阶段，
将参数替换为--dataset wpc和对应--data-root即可。
WPC文件若带_rounded后缀，原始文件查找时会识别别名，不会改名原文件。

但原历史WPC PNG的T-Net权重不可得。新训T-Net能形成完整运行流程，
不保证与历史PNG逐像素一致或产生完全相同指标；核对论文数值优先用第4节缓存路线。
不同机器、版本或重新生成的缓存也可能造成训练数值差异。

## 7. 训练设置、断点续训和结果

默认每折100轮、seed42、batch4、训练8patch/验证32patch；WPC除MOS100，
SJTU除MOS10且启用基础读取器的XYZRGB归一化。两套因素映射自动区分。
无需手工拼模型或修改训练loss。

输出在DFGC-Net/outputs/<dataset>/fold_N/version_0/：

- config.json：配置、基础源码与新增源码哈希、权重及划分哈希。
- checkpoints/：SROCC最佳top3及last.ckpt。
- predictions/epoch_XXX.csv：每轮的样本名、预测和MOS。
- completed.json：该折最佳epoch、权重路径及指标。
- TensorBoard事件文件：训练过程。

outputs/<dataset>/summary.csv汇总已完成折及算术平均；未跑齐时标记mean_available，
全部完成后标记mean_all_folds。不要把未跑齐的均值当成完整交叉验证成绩。
保留top3和last会占较多磁盘空间，运行前请预留存储。

中断后重新运行相同训练命令，自动从last.ckpt恢复，已完成折会跳过。
修改了配置、代码或上游版本时，需用--output指定新实验目录，防止覆盖/混用结果。
可用--folds 1 4选折，--workers 0关闭加载多进程。
恢复optimizer、scheduler、AMP/RNG状态不等于保证worker增强轨迹逐位不变。

## 8. 重新评价训练好的权重

读取该折completed.json中的checkpoint路径，用真实路径替换下方占位符：

```bash
python evaluate.py --upstream ../DA-Net --dataset wpc --data-root ../data/wpc --fold 1 --checkpoint "该折最佳权重.ckpt" --output ../results/wpc_fold1.csv
```

SJTU换成--dataset sjtu及其数据目录。评价输出逐样本CSV及.metrics.json。
SROCC/KROCC从原始预测和MOS计算，PLCC/RMSE沿用加载的基础指标实现；
原参考为无单调约束的五参数拟合，并额外保存plcc_raw。
训练仍按每轮留出折最高SROCC选模，没有额外内部验证集。

最终质量权重尚未提供下载入口，请先按上述步骤训练，再评价生成的最佳权重。
仅加载Swin预训练权重的模型不能用于报告最终质量评价成绩。
