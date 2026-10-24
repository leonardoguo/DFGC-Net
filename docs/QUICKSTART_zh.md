# 从哪里开始

[English reproduction guide](READER_GUIDE.md) | [中文复现指南](READER_GUIDE_zh.md)

完整操作手册见 [读者复现指南](READER_GUIDE_zh.md)。
常规读者无需修改DA-Net源码，也不需要手工编写训练器。

三个主要入口：
- prepare.py：元数据/划分、局部patch、全局六视图、T-Net、局部PNG和完整性检查。
- train.py：自动加载外部DA-Net、接入GTFS/LGQC、训练各折及断点续训。
- evaluate.py：加载某折的联合权重，重新计算预测和指标。

DA-Net放在本项目外，通过--upstream指定；数据通过--data-root指定。
下载项、放置目录、环境安装和逐条命令均已列在复现指南中。

本包分发新增模块和自己的运行编排代码，不分发基础骨干、T-Net架构、
局部投影实现、原有损失及指标算法；这些运行时从读者单独获得的DA-Net导入。
