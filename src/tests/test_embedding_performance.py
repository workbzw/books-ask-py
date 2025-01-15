import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class EmbeddingPerformanceTester:
    """用于测试不同embedding模型性能的测试类"""
    
    def __init__(self, model_names: List[str] = None):
        """
        初始化性能测试器
        
        Args:
            model_names: embedding模型名称列表，例如 ['all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2']
        """
        if model_names is None:
            model_names = ['all-MiniLM-L6-v2']  # 默认模型
        
        self.models = {}
        for name in model_names:
            try:
                self.models[name] = SentenceTransformer(name)
            except Exception as e:
                print(f"加载模型 {name} 失败: {str(e)}")
    
    def test_single_model(self, 
                         model_name: str, 
                         texts: List[str], 
                         batch_size: int = 32,
                         num_runs: int = 3) -> Dict:
        """
        测试单个模型的性能
        
        Args:
            model_name: 模型名称
            texts: 要编码的文本列表
            batch_size: 批处理大小
            num_runs: 重复运行次数
            
        Returns:
            包含性能指标的字典
        """
        model = self.models[model_name]
        total_time = 0
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            # 使用批处理进行编码
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                _ = model.encode(batch)
            
            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            total_time += run_time
        
        return {
            'model_name': model_name,
            'num_texts': len(texts),
            'batch_size': batch_size,
            'avg_time': total_time / num_runs,
            'min_time': min(times),
            'max_time': max(times),
            'times_per_text': (total_time / num_runs) / len(texts)
        }
    
    def run_benchmark(self, 
                     texts: List[str], 
                     batch_sizes: List[int] = None,
                     num_runs: int = 3) -> pd.DataFrame:
        """
        运行完整的基准测试
        
        Args:
            texts: 要编码的文本列表
            batch_sizes: 要测试的批处理大小列表
            num_runs: 每个配置重复运行的次数
            
        Returns:
            包含所有测试结果的DataFrame
        """
        if batch_sizes is None:
            batch_sizes = [16, 32, 64]
        
        results = []
        for model_name in tqdm(self.models, desc="Testing models"):
            for batch_size in batch_sizes:
                result = self.test_single_model(
                    model_name=model_name,
                    texts=texts,
                    batch_size=batch_size,
                    num_runs=num_runs
                )
                results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_results(self, results: pd.DataFrame, save_path: str = None):
        """
        可视化测试结果
        
        Args:
            results: 测试结果DataFrame
            save_path: 图表保存路径（可选）
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results, x='model_name', y='avg_time', hue='batch_size')
        plt.title('Embedding Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def main():
    # 示例用法
    # 准备测试数据
    test_texts = [
        "这是第一个测试文本",
        "这是第二个测试文本，比第一个长一些",
        "这是第三个测试文本，我们要测试embedding模型的性能"
    ] * 100  # 创建300个测试文本
    
    # 生成1000个字符的测试文本
    long_test_texts = [
        "这是一个测试文本，用来测试embedding模型性能",
        "我们需要生成一些长度适中的中文句子来测试",
        "自然语言处理模型需要合适的测试数据集", 
        "通过这些测试文本可以评估模型的效果表现",
        "中文分词和向量表示是重要的基础任务之一",
        "希望这些测试用例能够帮助改进模型效果",
    ] * 50  # 创建300个测试文本
    
    # 初始化测试器
    tester = EmbeddingPerformanceTester([
        'all-MiniLM-L6-v2',
        'BAAI/bge-base-zh'
    ])
    # 测试准确率
    def test_accuracy(self, texts: List[str], model_name: str) -> float:
        """
        测试embedding模型的准确率
        
        Args:
            texts: 测试文本列表
            model_name: 模型名称
            
        Returns:
            float: 准确率分数
        """
        model = SentenceTransformer(model_name)
        
        # 生成文本的embeddings
        embeddings = model.encode(texts)
        
        # 计算余弦相似度矩阵
        similarities = cosine_similarity(embeddings)
        
        # 计算准确率
        # 这里假设相似的文本应该有较高的相似度(>0.8)
        accuracy = 0
        total = 0
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if i != j:
                    # 检查文本相似度是否与预期一致
                    expected_similar = self._is_similar_text(texts[i], texts[j])
                    actual_similar = similarities[i][j] > 0.8
                    if expected_similar == actual_similar:
                        accuracy += 1
                    total += 1
                    
        return accuracy / total if total > 0 else 0
    
    def _is_similar_text(self, text1: str, text2: str) -> bool:
        """
        判断两个文本是否相似
        这里使用简单的启发式方法，可以根据具体需求改进
        """
        # 使用编辑距离判断文本相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity > 0.8
        
    # 测试各个模型的准确率
    print("\n测试模型准确率:")
    for model_name in tester.model_names:
        accuracy = tester.test_accuracy(test_texts[:10], model_name)
        print(f"{model_name}: {accuracy:.2%}")
    # 运行基准测试
    results = tester.run_benchmark(
        texts=test_texts,
        batch_sizes=[16, 32, 64],
        num_runs=3
    )
    
    print("\n性能测试结果:")
    print(results)
    
    # 绘制结果图表
    tester.plot_results(results, 'embedding_performance.png')
    
    # 运行1000字符的基准测试
    results_long = tester.run_benchmark(
        texts=long_test_texts, 
        batch_sizes=[16, 32, 64],
        num_runs=3
    )
    
    print("\n1000字符文本的性能测试结果:")
    print(results_long)
    
    # 绘制1000字符的结果图表
    tester.plot_results(results_long, 'embedding_performance_long.png')

if __name__ == "__main__":
    main() 