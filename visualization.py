#!/usr/bin/env python3
"""
GRACE项目完整可视化脚本
包含：
1. 性能指标可视化
2. t-SNE/UMAP降维可视化
3. 梯度激活图（Saliency Map）
4. Attention权重可视化
5. 架构流程图

使用方法:
    python visualization.py --all  # 生成所有可视化
    python visualization.py --tsne  # 仅生成t-SNE可视化
    python visualization.py --saliency  # 仅生成梯度激活图
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F

# ========== 配置HuggingFace镜像源 ==========
# 在导入sentence_transformers之前设置环境变量
HF_MIRRORS = [
    "https://hf-mirror.com",  # 官方镜像
    "https://huggingface.co",  # 原始地址
]

# 设置环境变量以使用镜像源
os.environ.setdefault("HF_ENDPOINT", HF_MIRRORS[0])
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# 设置sentence-transformers使用镜像源
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(Path.home() / ".cache" / "sentence_transformers"))

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# 尝试导入降维库
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    print("⚠️ 未安装sklearn，t-SNE功能不可用")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ 未安装umap-learn，UMAP功能不可用")

class GraceVisualizer:
    """GRACE可视化生成器"""
    
    def __init__(self, project_root: Path = None):
        """
        初始化可视化器
        
        Args:
            project_root: 项目根目录
        """
        if project_root is None:
            self.project_root = Path(__file__).parent
        else:
            self.project_root = Path(project_root)
        
        self.figs_dir = self.project_root / "figs"
        self.output_dir = self.project_root / "outputs"
        self.log_dir = self.project_root / "log"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # 确保目录存在
        self.figs_dir.mkdir(exist_ok=True)
        
        # 从日志文件解析评估结果
        self.metrics_data = self._parse_metrics_from_logs()
        
        print(f"📊 可视化输出目录: {self.figs_dir}")
        print(f"📈 已解析 {len(self.metrics_data)} 个评估结果")
        print(f"🌐 HuggingFace镜像源: {os.environ.get('HF_ENDPOINT', '未设置')}")
    
    def _parse_metrics_from_logs(self) -> Dict:
        """从日志文件解析评估指标"""
        metrics = {}
        
        # 解析CodeBERT结果
        codebert_logs = {
            "bigvul": self.output_dir / "bigvulmetricscodebert-base.log",
            "reveal": self.output_dir / "revealmetricscodebert-base.log",
            "devign": self.output_dir / "devignmetricscodebert-base.log"
        }
        
        for dataset, log_file in codebert_logs.items():
            if log_file.exists():
                metrics[f"{dataset}_codebert"] = self._parse_log_file(log_file, "codebert")
        
        # 解析GPT-4结果
        gpt4_logs = {
            "bigvul": self.log_dir / "bigvulmetricsgpt4.log",
            "reveal": self.log_dir / "revealmetricsgpt4.log",
            "devign": self.log_dir / "devignmetricsgpt4.log"
        }
        
        for dataset, log_file in gpt4_logs.items():
            if log_file.exists():
                metrics[f"{dataset}_gpt4"] = self._parse_log_file(log_file, "gpt4")
        
        return metrics
    
    def _parse_log_file(self, log_file: Path, model_type: str) -> Dict:
        """解析单个日志文件"""
        metrics = {
            "model": model_type,
            "dataset": "",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_samples": 0
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析数据集名称
            dataset_match = re.search(r'Dataset:\s*(\w+)', content)
            if dataset_match:
                metrics["dataset"] = dataset_match.group(1).lower()
            
            # 解析指标
            patterns = {
                "accuracy": r'Accuracy[:\s]+([\d.]+)',
                "precision": r'Precision[:\s]+([\d.]+)',
                "recall": r'Recall[:\s]+([\d.]+)',
                "f1_score": r'F1[-\s]?Score[:\s]+([\d.]+)',
                "total_samples": r'Total Samples[:\s]+(\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if key == "total_samples":
                        metrics[key] = int(value)
                    else:
                        metrics[key] = float(value)
        
        except Exception as e:
            print(f"⚠️ 解析日志文件失败 {log_file}: {e}")
        
        return metrics
    
    def plot_tsne_umap_visualization(self, dataset_name: str = "bigvul", 
                                     max_samples: int = 1000,
                                     save_path: str = None):
        """
        绘制t-SNE和UMAP降维可视化
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大样本数（用于加速）
            save_path: 保存路径
        """
        print(f"\n🔄 开始生成t-SNE/UMAP可视化 ({dataset_name})...")
        
        # 加载数据和嵌入
        embeddings, labels = self._load_embeddings_and_labels(dataset_name, max_samples)
        
        if embeddings is None or len(embeddings) == 0:
            print(f"⚠️ 无法加载 {dataset_name} 的嵌入数据")
            print("💡 提示: 需要先运行评估以生成嵌入数据")
            return
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Code Embedding Visualization - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # t-SNE可视化
        if HAS_TSNE:
            print("  📊 计算t-SNE降维...")
            try:
                # 新版本sklearn使用max_iter，旧版本使用n_iter
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
                except TypeError:
                    # 兼容旧版本
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                
                embeddings_tsne = tsne.fit_transform(embeddings)
                
                ax1 = axes[0]
                scatter1 = ax1.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                                      c=labels, cmap='RdYlBu_r', alpha=0.6, s=20)
                ax1.set_title('t-SNE Visualization', fontsize=13, fontweight='bold')
                ax1.set_xlabel('t-SNE Dimension 1', fontsize=11)
                ax1.set_ylabel('t-SNE Dimension 2', fontsize=11)
                ax1.grid(True, alpha=0.3)
                plt.colorbar(scatter1, ax=ax1, label='Vulnerability Label')
            except Exception as e:
                print(f"  ⚠️ t-SNE计算失败: {e}")
                axes[0].text(0.5, 0.5, f't-SNE计算失败\n{str(e)[:50]}', 
                            ha='center', va='center', fontsize=10)
                axes[0].set_title('t-SNE Visualization', fontsize=13, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 't-SNE not available\n(install sklearn)', 
                        ha='center', va='center', fontsize=12)
            axes[0].set_title('t-SNE Visualization', fontsize=13, fontweight='bold')
        
        # UMAP可视化
        if HAS_UMAP:
            print("  📊 计算UMAP降维...")
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                embeddings_umap = reducer.fit_transform(embeddings)
                
                ax2 = axes[1]
                scatter2 = ax2.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                                      c=labels, cmap='RdYlBu_r', alpha=0.6, s=20)
                ax2.set_title('UMAP Visualization', fontsize=13, fontweight='bold')
                ax2.set_xlabel('UMAP Dimension 1', fontsize=11)
                ax2.set_ylabel('UMAP Dimension 2', fontsize=11)
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter2, ax=ax2, label='Vulnerability Label')
            except Exception as e:
                print(f"  ⚠️ UMAP计算失败: {e}")
                axes[1].text(0.5, 0.5, f'UMAP计算失败\n{str(e)[:50]}', 
                            ha='center', va='center', fontsize=10)
                axes[1].set_title('UMAP Visualization', fontsize=13, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'UMAP not available\n(install umap-learn)', 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('UMAP Visualization', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figs_dir / f"tsne_umap_{dataset_name}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE/UMAP可视化已保存: {save_path}")
        plt.close()
    
    def _load_embeddings_and_labels(self, dataset_name: str, max_samples: int = 1000):
        """
        加载代码嵌入和标签
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大样本数
            
        Returns:
            (embeddings, labels): 嵌入矩阵和标签列表
        """
        try:
            # 尝试从处理后的数据文件加载
            data_file = self.data_dir / f"{dataset_name}_test_processed.json"
            if not data_file.exists():
                # 尝试其他路径
                from config.config import Config
                config = Config()
                data_file = config.get_processed_dataset_path(dataset_name, "test")
            
            if not data_file.exists():
                print(f"⚠️ 数据文件不存在: {data_file}")
                return None, None
            
            # 加载数据
            import pandas as pd
            df = pd.read_json(data_file, orient='records', lines=True)
            
            if len(df) == 0:
                return None, None
            
            # 限制样本数
            if len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=42)
            
            # 提取代码和标签
            codes = df['code'].tolist() if 'code' in df.columns else []
            labels = df['label'].tolist() if 'label' in df.columns else []

            
            # ===== 添加诊断代码 =====
            if labels:
                labels_array = np.array(labels)
                print(f"  📊 标签统计信息:")
                print(f"     标签总数: {len(labels)}")
                print(f"     唯一值: {np.unique(labels_array)}")
                print(f"     最小值: {labels_array.min()}")
                print(f"     最大值: {labels_array.max()}")
                print(f"     平均值: {labels_array.mean():.4f}")
                print(f"     标签0的数量: {np.sum(labels_array == 0)}")
                print(f"     标签1的数量: {np.sum(labels_array == 1)}")
                print(f"     标签类型: {type(labels[0])}")
                print(f"     标签示例: {labels[:10]}")
                
                # 检查是否有问题
                if len(np.unique(labels_array)) == 1:
                    print(f"  ⚠️ 警告: 所有标签都是同一个值！")
                if labels_array.min() == labels_array.max() == 0:
                    print(f"  ⚠️ 警告: 所有标签都是0，可能是数据加载问题！")
            else:
                print(f"  ⚠️ 警告: 没有找到标签数据！")
                labels = [0] * len(codes)  # 默认值
            
            if not codes:
                return None, None
            
            # 使用模型生成嵌入（需要导入模型）
            print(f"  🔄 生成 {len(codes)} 个代码样本的嵌入...")
            embeddings = self._generate_embeddings(codes)
            
            return embeddings, labels
            
        except Exception as e:
            print(f"⚠️ 加载嵌入数据失败: {e}")
            return None, None
    
    def _generate_embeddings(self, codes: List[str]) -> np.ndarray:
        """
        生成代码嵌入（优先使用项目中的CodeBERT模型）
        
        Args:
            codes: 代码列表
            
        Returns:
            嵌入矩阵
        """
        print(f"  🔄 尝试使用项目中的CodeBERT模型生成嵌入...")
        
        try:
            # 优先使用项目中的CodeBERT模型
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            from config.config import Config
            config = Config()
            model_name = config.model_name  # microsoft/codebert-base
            
            print(f"  📥 加载CodeBERT模型: {model_name}")
            
            # 检查本地是否有模型
            model_path = config.get_model_path(model_name)
            if model_path.exists() and any(model_path.iterdir()):
                print(f"  ✅ 使用本地模型: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModel.from_pretrained(str(model_path))
            else:
                print(f"  📥 从HuggingFace加载模型（使用镜像源）...")
                # 设置镜像源
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            # 批量编码
            embeddings_list = []
            batch_size = 16
            
            print(f"  🔄 编码 {len(codes)} 个代码样本...")
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_codes,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                # 生成嵌入
                with torch.no_grad():
                    outputs = model(**inputs)
                    # 使用CLS token的嵌入（第一个token）
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings_list.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"    进度: {i+batch_size}/{len(codes)}")
            
            embeddings = np.vstack(embeddings_list)
            print(f"  ✅ 成功生成嵌入，形状: {embeddings.shape}")
            return embeddings.astype('float32')
            
        except Exception as e:
            print(f"  ⚠️ 使用CodeBERT失败: {e}")
            print(f"  💡 尝试使用sentence-transformers作为备选...")
            
            # 备选方案：使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                
                # 正确的模型名称（不带前缀）
                model_name = 'all-MiniLM-L6-v2'
                
                # 检查本地缓存
                cache_dir = Path.home() / ".cache" / "sentence_transformers" / model_name
                if cache_dir.exists() and any(cache_dir.iterdir()):
                    print(f"  📦 使用本地sentence-transformers模型")
                    model = SentenceTransformer(str(cache_dir))
                else:
                    # 设置镜像源
                    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    print(f"  📥 从镜像源下载: {model_name}")
                    model = SentenceTransformer(model_name)
                
                embeddings = model.encode(codes, show_progress_bar=True, 
                                         batch_size=32, convert_to_numpy=True)
                print(f"  ✅ 使用sentence-transformers生成嵌入成功")
                return embeddings.astype('float32')
                
            except Exception as e2:
                print(f"  ⚠️ sentence-transformers也失败: {e2}")
                print(f"  💡 使用随机嵌入作为占位符（仅用于可视化框架）")
                # 返回随机嵌入作为占位符（CodeBERT是768维）
                return np.random.randn(len(codes), 768).astype('float32')
    
    def plot_saliency_map(self, code: str, model_path: str = None, 
                     save_path: str = None):
        """
        绘制梯度激活图（Saliency Map）
        
        Args:
            code: 代码片段
            model_path: 模型路径
            save_path: 保存路径
        """
        print("\n🔄 开始生成梯度激活图...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            import torch.nn.functional as F
            
            # 检查PyTorch版本
            torch_version = torch.__version__
            print(f"  📦 PyTorch版本: {torch_version}")
            
            # 加载模型
            if model_path is None:
                from config.config import Config
                config = Config()
                model_name = config.model_name
            else:
                model_name = model_path
            
            print(f"  📥 加载模型: {model_name}")
            
            # 检查本地模型路径
            from config.config import Config
            config = Config()
            local_model_path = config.get_model_path(model_name)
            
            try:
                if local_model_path.exists() and any(local_model_path.iterdir()):
                    print(f"  ✅ 使用本地模型: {local_model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
                    model = AutoModel.from_pretrained(str(local_model_path))
                else:
                    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
            except Exception as e:
                if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                    print(f"  ⚠️ PyTorch版本问题: {e}")
                    print(f"  💡 解决方案: 升级PyTorch或使用safetensors格式")
                    return
                else:
                    raise e
            
            device = next(model.parameters()).device
            
            # ===== 关键修复：临时切换到训练模式以启用梯度 =====
            was_training = model.training
            model.train()  # 切换到训练模式以启用梯度计算
            
            try:
                # 准备输入
                tokens = tokenizer(code, return_tensors="pt", truncation=True, 
                                max_length=512, padding=True)
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                
                # ===== 关键修复：使用嵌入层来计算梯度 =====
                # 获取嵌入层
                embedding_layer = model.embeddings.word_embeddings
                embeddings = embedding_layer(input_ids)
                
                # 关键：保留非叶子张量的梯度
                embeddings.retain_grad()
                
                # 前向传播（使用嵌入而不是input_ids）
                outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                
                # 计算损失（使用CLS token的嵌入）
                cls_embedding = hidden_states[:, 0, :]
                # 使用L2范数作为损失，这样更有意义
                loss = cls_embedding.norm(dim=1).sum()
                
                # 反向传播
                model.zero_grad()
                loss.backward()
                
                # 检查梯度是否存在
                if embeddings.grad is None:
                    print("  ⚠️ 警告: 无法获取梯度，尝试使用输入梯度...")
                    # 备选方案：使用input_ids的梯度
                    input_ids.requires_grad = True
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = outputs.last_hidden_state[:, 0, :].norm(dim=1).sum()
                    model.zero_grad()
                    loss.backward()
                    
                    if input_ids.grad is None:
                        print("  ⚠️ 无法获取任何梯度，跳过梯度激活图生成")
                        return
                    else:
                        gradients = input_ids.grad.abs()
                        saliency_scores = gradients.squeeze().cpu().numpy()
                else:
                    # 获取嵌入的梯度
                    gradients = embeddings.grad.abs()
                    # 对每个token的所有维度求平均，得到每个token的重要性
                    saliency_scores = gradients.mean(dim=2).squeeze().cpu().numpy()
                
                # 获取token文本
                token_ids = input_ids.squeeze().cpu().numpy()
                token_texts = []
                for tid in token_ids:
                    token_text = tokenizer.decode([tid])
                    # 清理特殊token的显示
                    if token_text.startswith('<') and token_text.endswith('>'):
                        token_text = token_text.replace('<', '[').replace('>', ']')
                    token_texts.append(token_text)
                
                # 创建可视化
                fig, ax = plt.subplots(figsize=(14, max(8, len(token_texts) * 0.3)))
                
                # 绘制saliency scores
                y_pos = np.arange(len(token_texts))
                if saliency_scores.max() > 0:
                    colors = plt.cm.Reds(saliency_scores / saliency_scores.max())
                else:
                    colors = plt.cm.Reds(np.zeros_like(saliency_scores))
                
                bars = ax.barh(y_pos, saliency_scores, color=colors, alpha=0.7)
                
                # 设置标签
                ax.set_yticks(y_pos)
                ax.set_yticklabels(token_texts, fontsize=8)
                ax.set_xlabel('Saliency Score', fontsize=12, fontweight='bold')
                ax.set_title('Gradient-based Saliency Map', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # 添加颜色条
                if saliency_scores.max() > 0:
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                            norm=plt.Normalize(vmin=0, vmax=saliency_scores.max()))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Saliency Score', fontsize=11)
                
                plt.tight_layout()
                
                if save_path is None:
                    save_path = self.figs_dir / "saliency_map.png"
                
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 梯度激活图已保存: {save_path}")
                plt.close()
                
            finally:
                # 恢复原来的模式
                if not was_training:
                    model.eval()
                
        except Exception as e:
            print(f"⚠️ 生成梯度激活图失败: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_attention_weights(self, code: str, model_path: str = None,
                           save_path: str = None):
        """
        绘制Attention权重可视化
        
        Args:
            code: 代码片段
            model_path: 模型路径
            save_path: 保存路径
        """
        print("\n🔄 开始生成Attention权重可视化...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # 检查PyTorch版本
            import torch
            torch_version = torch.__version__
            print(f"  📦 PyTorch版本: {torch_version}")
            
            # 加载模型
            if model_path is None:
                from config.config import Config
                config = Config()
                model_name = config.model_name
            else:
                model_name = model_path
            
            print(f"  📥 加载模型: {model_name}")
            
            # 检查本地模型路径
            from config.config import Config
            config = Config()
            local_model_path = config.get_model_path(model_name)
            
            try:
                if local_model_path.exists() and any(local_model_path.iterdir()):
                    print(f"  ✅ 使用本地模型: {local_model_path}")
                    # 尝试使用safetensors加载
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            str(local_model_path),
                            use_safetensors=True
                        )
                        model = AutoModel.from_pretrained(
                            str(local_model_path),
                            use_safetensors=True,
                            output_attentions=True
                        )
                    except (Exception, TypeError) as e:
                        # 如果safetensors失败或参数不支持，尝试普通加载
                        print(f"  ⚠️ safetensors加载失败，使用普通加载: {e}")
                        tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
                        model = AutoModel.from_pretrained(
                            str(local_model_path),
                            output_attentions=True
                        )
                else:
                    # 从HuggingFace加载
                    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModel.from_pretrained(
                            model_name, 
                            use_safetensors=True,
                            output_attentions=True
                        )
                    except (Exception, TypeError) as e:
                        # 如果safetensors参数不支持，使用普通加载
                        print(f"  ⚠️ safetensors参数不支持，使用普通加载: {e}")
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModel.from_pretrained(
                            model_name,
                            output_attentions=True
                        )
            except Exception as e:
                if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                    print(f"  ⚠️ PyTorch版本问题: {e}")
                    print(f"  💡 解决方案:")
                    print(f"     1. 升级PyTorch: pip install torch>=2.6.0")
                    print(f"     2. 或使用safetensors格式的模型")
                    print(f"     3. 或跳过此可视化")
                    return
                else:
                    raise e
            
            model.eval()
            device = next(model.parameters()).device
            
            # 准备输入
            tokens = tokenizer(code, return_tensors="pt", truncation=True, 
                            max_length=128, padding=True)
            input_ids = tokens['input_ids'].to(device)
            
            # 前向传播并获取attention
            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_attentions=True)
                attentions = outputs.attentions  # 所有层的attention
            
            # 使用最后一层的attention
            attention = attentions[-1].squeeze().cpu().numpy()
            
            # 平均所有注意力头（如果有多个头）
            if len(attention.shape) == 3:
                attention = attention.mean(axis=0)  # 平均所有头
            
            # 获取token文本
            token_ids = input_ids.squeeze().cpu().numpy()
            token_texts = [tokenizer.decode([tid]) for tid in token_ids]
            
            # 创建热力图
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(attention, cmap='Blues', aspect='auto')
            
            # 设置标签
            ax.set_xticks(np.arange(len(token_texts)))
            ax.set_yticks(np.arange(len(token_texts)))
            ax.set_xticklabels(token_texts, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(token_texts, fontsize=8)
            
            ax.set_xlabel('Key Position', fontsize=12, fontweight='bold')
            ax.set_ylabel('Query Position', fontsize=12, fontweight='bold')
            ax.set_title('Attention Weights Heatmap (Last Layer)', 
                        fontsize=14, fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Attention Weight')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.figs_dir / "attention_weights.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Attention权重可视化已保存: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 生成Attention可视化失败: {e}")
            if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                print("💡 提示: 需要升级PyTorch到v2.6+或使用safetensors格式")
            import traceback
            traceback.print_exc()
    
    def plot_attention_weights(self, code: str, model_path: str = None,
                               save_path: str = None):
        """
        绘制Attention权重可视化
        
        Args:
            code: 代码片段
            model_path: 模型路径
            save_path: 保存路径
        """
        print("\n🔄 开始生成Attention权重可视化...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # 加载模型
            if model_path is None:
                from config.config import Config
                config = Config()
                model_name = config.model_name
            else:
                model_name = model_path
            
            print(f"  📥 加载模型: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            # 准备输入
            tokens = tokenizer(code, return_tensors="pt", truncation=True, 
                            max_length=128, padding=True)
            input_ids = tokens['input_ids']
            
            # 前向传播并获取attention
            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_attentions=True)
                attentions = outputs.attentions  # 所有层的attention
            
            # 使用最后一层的attention
            attention = attentions[-1].squeeze().cpu().numpy()
            
            # 获取token文本
            token_ids = input_ids.squeeze().cpu().numpy()
            token_texts = [tokenizer.decode([tid]) for tid in token_ids]
            
            # 创建热力图
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(attention, cmap='Blues', aspect='auto')
            
            # 设置标签
            ax.set_xticks(np.arange(len(token_texts)))
            ax.set_yticks(np.arange(len(token_texts)))
            ax.set_xticklabels(token_texts, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(token_texts, fontsize=8)
            
            ax.set_xlabel('Key Position', fontsize=12, fontweight='bold')
            ax.set_ylabel('Query Position', fontsize=12, fontweight='bold')
            ax.set_title('Attention Weights Heatmap (Last Layer)', 
                        fontsize=14, fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Attention Weight')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.figs_dir / "attention_weights.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Attention权重可视化已保存: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 生成Attention可视化失败: {e}")
    
    def plot_metrics_comparison(self, save_path: str = None):
        """绘制性能指标对比图"""
        if not self.metrics_data:
            print("⚠️ 没有可用的评估数据")
            return
        
        # 准备数据
        datasets = ["bigvul", "reveal", "devign"]
        models = ["codebert", "gpt4"]
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GRACE Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # 收集数据
            data = []
            labels = []
            for dataset in datasets:
                for model in models:
                    key = f"{dataset}_{model}"
                    if key in self.metrics_data:
                        value = self.metrics_data[key].get(metric, 0.0)
                        data.append(value)
                        labels.append(f"{dataset.upper()}\n{model.upper()}")
            
            if data:
                bars = ax.bar(range(len(data)), data, 
                             color=sns.color_palette("husl", len(data)),
                             alpha=0.8, edgecolor='black', linewidth=1.5)
                
                for i, (bar, val) in enumerate(zip(bars, data)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
                ax.set_ylim(0, max(data) * 1.2 if data else 1)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                           fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figs_dir / "metrics_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 性能指标对比图已保存: {save_path}")
        plt.close()
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        print("\n" + "="*60)
        print("开始生成GRACE可视化图表")
        print("="*60 + "\n")
        
        # 基础可视化
        self.plot_metrics_comparison()
        
        # 降维可视化
        for dataset in ["bigvul", "reveal", "devign"]:
            try:
                self.plot_tsne_umap_visualization(dataset, max_samples=500)
            except Exception as e:
                print(f"⚠️ {dataset} 降维可视化失败: {e}")
        
        # 示例代码的梯度激活图
        example_code = """
        void vulnerable_function(char *input) {
            char buffer[10];
            strcpy(buffer, input);  // Buffer overflow vulnerability
            printf("%s", buffer);
        }
        """
        try:
            self.plot_saliency_map(example_code)
        except Exception as e:
            print(f"⚠️ 梯度激活图生成失败: {e}")
        
        # Attention权重可视化
        try:
            self.plot_attention_weights(example_code)
        except Exception as e:
            print(f"⚠️ Attention可视化失败: {e}")
        
        print("\n" + "="*60)
        print("✅ 所有可视化图表生成完成！")
        print(f"📁 输出目录: {self.figs_dir}")
        print("="*60 + "\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GRACE可视化工具")
    parser.add_argument("--all", action="store_true", help="生成所有可视化")
    parser.add_argument("--tsne", action="store_true", help="生成t-SNE/UMAP可视化")
    parser.add_argument("--saliency", action="store_true", help="生成梯度激活图")
    parser.add_argument("--attention", action="store_true", help="生成Attention可视化")
    parser.add_argument("--metrics", action="store_true", help="生成性能指标图")
    parser.add_argument("--dataset", type=str, default="bigvul", 
                       help="数据集名称 (bigvul/reveal/devign)")
    parser.add_argument("--code", type=str, help="要分析的代码片段")
    parser.add_argument("--project-root", type=str, help="项目根目录路径")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = GraceVisualizer(project_root=args.project_root)
    
    # 根据参数生成可视化
    if args.all:
        visualizer.generate_all_visualizations()
    else:
        if args.tsne:
            visualizer.plot_tsne_umap_visualization(args.dataset)
        if args.saliency:
            code = args.code or """
            void vulnerable_function(char *input) {
                char buffer[10];
                strcpy(buffer, input);
            }
            """
            visualizer.plot_saliency_map(code)
        if args.attention:
            code = args.code or """
            void vulnerable_function(char *input) {
                char buffer[10];
                strcpy(buffer, input);
            }
            """
            visualizer.plot_attention_weights(code)
        if args.metrics:
            visualizer.plot_metrics_comparison()
        
        # 如果没有指定任何选项，默认生成所有
        if not any([args.tsne, args.saliency, args.attention, args.metrics]):
            print("未指定选项，生成所有可视化...")
            visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()