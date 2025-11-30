#!/usr/bin/env python
"""
示例训练脚本

演示如何使用生成的雷达RFF数据集进行模型训练

任务：设备识别（20分类）

作者：根据研究需求生成
日期：2025-10-18
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# 数据集类
# ============================================================================

class RadarRFFDataset(Dataset):
    """雷达RFF数据集"""
    
    def __init__(self, h5_file_path, task='device', transform=None):
        """
        参数:
            h5_file_path: HDF5文件路径
            task: 任务类型 ('device', 'modulation', 'combined')
            transform: 数据变换（可选）
        """
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.iq_data = self.h5_file['iq_data']
        self.device_labels = self.h5_file['device_labels'][:]
        self.mod_labels = self.h5_file['modulation_labels'][:]
        self.combined_labels = self.h5_file['combined_labels'][:]
        self.task = task
        self.transform = transform
    
    def __len__(self):
        return self.iq_data.shape[0]
    
    def __getitem__(self, idx):
        # 读取IQ数据
        iq = self.iq_data[idx]
        
        # 转换为2通道输入 (I, Q)
        i_channel = np.real(iq).astype(np.float32)
        q_channel = np.imag(iq).astype(np.float32)
        x = np.stack([i_channel, q_channel], axis=0)  # (2, 4096)
        
        # 选择标签
        if self.task == 'device':
            y = self.device_labels[idx]
        elif self.task == 'modulation':
            y = self.mod_labels[idx]
        else:  # combined
            y = self.combined_labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


# ============================================================================
# 简单的CNN模型
# ============================================================================

class SimpleCNN(nn.Module):
    """简单的1D CNN用于信号分类"""
    
    def __init__(self, num_classes=20, num_channels=2):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            # Conv Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            # Conv Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            # Conv Block 4
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# ============================================================================
# 训练和评估函数
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='训练')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / total, 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='验证'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100. * correct / total


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='雷达RFF数据集训练示例')
    parser.add_argument('--dataset', type=str, required=True,
                       help='HDF5数据集文件路径')
    parser.add_argument('--task', type=str, default='device',
                       choices=['device', 'modulation', 'combined'],
                       help='任务类型')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批量大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备ID (-1表示使用CPU)')
    parser.add_argument('--output-dir', type=str, default='./training_output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 确定类别数
    if args.task == 'device':
        num_classes = 20
    elif args.task == 'modulation':
        num_classes = 26
    else:  # combined
        num_classes = 520
    
    print(f"\n任务: {args.task} 识别")
    print(f"类别数: {num_classes}")
    
    # 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    full_dataset = RadarRFFDataset(args.dataset, task=args.task)
    
    # 划分训练集和验证集 (80%-20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    
    # 训练循环
    print(f"\n开始训练 (共 {args.epochs} 轮)...")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印摘要
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    
    print("\n" + "="*80)
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print("="*80)
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='训练')
    ax1.plot(history['val_loss'], label='验证')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(history['train_acc'], label='训练')
    ax2.plot(history['val_acc'], label='验证')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), 
               dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存至: {args.output_dir}/training_curves.png")
    
    # 保存训练历史
    np.save(os.path.join(args.output_dir, 'history.npy'), history)
    print(f"✅ 训练历史已保存至: {args.output_dir}/history.npy")


if __name__ == "__main__":
    main()

