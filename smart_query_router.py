"""
Smart Query Router with LoRA-based SLM Selection
实现基于embedding相似度的智能查询路由系统
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import openai
import atexit



class SmartQueryRouter:
    """
    智能查询路由器：根据query的embedding选择最合适的领域专家SLM
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
        device: str | None = None,
        is_llm_involved=True
    ):
        """
        初始化路由器
        
        Args:
            embedding_model_name: 用于生成embedding的模型名称
            similarity_threshold: 相似度阈值，低于此值将使用Web搜索
            device: 计算设备 (cuda/cpu)
        """
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 加载embedding模型
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(self.device)
        
        # 存储SLM配置
        self.slm_configs = {}
        self.slm_embeddings = {}
        self.loaded_models = {}
        self.is_llm_involved = is_llm_involved
        
        # 相似度阈值
        self.similarity_threshold = similarity_threshold
        atexit.register(self.exit)

    def exit(self):
        self.unload_all_slms()
        
    def register_slm(
        self,
        domain_name: str,
        base_model_id: str,
        lora_weights_path: str,
        domain_description: str
    ):
        """
        注册一个领域专家SLM
        
        Args:
            domain_name: 领域名称 (e.g., "medical", "finance", "legal", "tech")
            base_model_id: 基础模型的HuggingFace ID
            lora_weights_path: LoRA权重路径
            domain_description: 领域描述，用于生成domain embedding
        """
        print(f"Registering SLM for domain: {domain_name}")
        
        self.slm_configs[domain_name] = {
            "base_model_id": base_model_id,
            "lora_weights_path": lora_weights_path,
            "domain_description": domain_description
        }
        
        # 生成domain embedding (基于领域描述)
        domain_embedding = self.embedding_model.encode(
            domain_description,
            convert_to_tensor=True,
            device=self.device
        )
        self.slm_embeddings[domain_name] = domain_embedding
        
        print(f"Domain '{domain_name}' registered successfully")

    def register_slm_v2(
            self,
            domain_name: str,
            base_model_id: str,
            lora_weights_path: str,
            domain_description_list: list
        ):
            """
            注册一个领域专家SLM
            
            Args:
                domain_name: 领域名称 (e.g., "medical", "finance", "legal", "tech")
                base_model_id: 基础模型的HuggingFace ID
                lora_weights_path: LoRA权重路径
                domain_description: 领域描述，用于生成domain embedding
            """
            print(f"Registering SLM for domain: {domain_name}")

            self.slm_configs[domain_name] = {
                "base_model_id": base_model_id,
                "lora_weights_path": lora_weights_path,
                "domain_description": domain_description_list
            }
            
            # 生成domain embedding (基于领域描述)
            domain_embedding = [self.embedding_model.encode(
                domain_d,
                convert_to_tensor=True,
                device=self.device
            ) for domain_d in domain_description_list]

            self.slm_embeddings[domain_name] = domain_embedding
            
            print(f"Domain '{domain_name}' registered successfully")
        
    def load_slm(self, domain_name: str):
        """
        按需加载SLM模型（包含LoRA权重）
        
        Args:
            domain_name: 要加载的领域名称
        """
        if domain_name in self.loaded_models:
            return self.loaded_models[domain_name]
        
        if domain_name not in self.slm_configs:
            raise ValueError(f"Domain '{domain_name}' not registered")
        
        config = self.slm_configs[domain_name]
        print(f"Loading SLM for domain: {domain_name}")
        
        try:
            # 加载base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config["base_model_id"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["base_model_id"],
                trust_remote_code=True
            )
            
            # 加载LoRA权重
            if os.path.exists(config["lora_weights_path"]):
                model = PeftModel.from_pretrained(
                    base_model,
                    config["lora_weights_path"]
                )
                print(f"LoRA weights loaded from: {config['lora_weights_path']}")
            else:
                print(f"Warning: LoRA weights not found at {config['lora_weights_path']}, using base model only")
                model = base_model
            
            # 设置为评估模式
            model.eval()
            
            self.loaded_models[domain_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            
            print(f"SLM for '{domain_name}' loaded successfully")
            return self.loaded_models[domain_name]
            
        except Exception as e:
            print(f"Error loading SLM for '{domain_name}': {e}")
            raise
    
    def get_query_embedding(self, query: str) -> torch.Tensor:
        """
        将query转换为embedding
        
        Args:
            query: 用户输入的查询
            
        Returns:
            query的embedding tensor
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        return query_embedding
    
    def calculate_similarity(
        self,
        query_embedding: torch.Tensor,
        domain_embedding: torch.Tensor
    ) -> float:
        """
        计算query embedding和domain embedding的余弦相似度
        
        Args:
            query_embedding: query的embedding
            domain_embedding: domain的embedding
            
        Returns:
            相似度分数 (0-1)
        """
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            domain_embedding.unsqueeze(0)
        ).item()
        
        return similarity
    
    def normalized(self, similarity: dict) -> dict[str, float]:
        # print(similarity)
        min_val = min(similarity.values())
        min_val = min(min_val, 0)
        similarity = {k: v + min_val + 0.00001 for k, v in similarity.items()} # 0.01 to avoid 0 in prob.

        sum_val = sum(similarity.values())

        return {k: v / sum_val for k, v in similarity.items()}

    
    def select_best_slm(self, query: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        选择最适合处理query的SLM
        
        Args:
            query: 用户输入的查询
            
        Returns:
            (选中的domain名称, 最高相似度分数, 所有domain的相似度分数字典)
        """
        # 步骤1: 获取query embedding
        print(f"\n[Step 1] Encoding query to embedding...")
        query_embedding = self.get_query_embedding(query)
        
        # 步骤2 & 3: 计算与所有domain的相似度
        print(f"[Step 2&3] Calculating similarity with all registered SLMs...")
        similarities = {}
        
        for domain_name, domain_embedding in self.slm_embeddings.items():
            similarity = self.calculate_similarity(query_embedding, domain_embedding)
            similarities[domain_name] = similarity

            print(f'{query=}, {domain_name=}, {similarity=}')

        # 找出最高相似度
        if not similarities:
            return None, 0.0, {}

        similarities = self.normalized(similarities)
        for domain_name, similarity in similarities.items():
            print(f"  - {domain_name}: {similarity:.4f}")

        # Find best domain expert regarding similarity
        best_domain, best_similarity = max(similarities.items(), key=lambda x: x[1])

        assert best_domain is not None
        best_similarity = similarities[best_domain]
        
        # 判断是否超过阈值
        if best_similarity < self.similarity_threshold:
            print(f"\n[Selection] Best similarity ({best_similarity:.4f}) below threshold ({self.similarity_threshold})")
            print(f"[Selection] Will use Web Search instead")
            return None, best_similarity, similarities
        
        print(f"\n[Selection] Selected domain: {best_domain} (similarity: {best_similarity:.4f})")
        return best_domain, best_similarity, similarities
    
    def remove_unrelated(self, similarity: dict[str, int]):
        result = {}

        for k, sim in similarity.items():
            if sim < 0.8:
                result[k] = 0
                continue

            result[k] = sim

        return result
    
    def select_best_slm_v2(self, query: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        选择最适合处理query的SLM
        
        Args:
            query: 用户输入的查询
            
        Returns:
            (选中的domain名称, 最高相似度分数, 所有domain的相似度分数字典)
        """
        # 步骤1: 获取query embedding
        print(f"\n[Step 1] Encoding query to embedding...")
        query_embedding = self.get_query_embedding(query)
        
        # 步骤2 & 3: 计算与所有domain的相似度
        print(f"[Step 2&3] Calculating similarity with all registered SLMs...")
        similarities = {}

        def calculate_similarity_v2(
            query_embedding: torch.Tensor,
            domain_embedding_list: list[torch.Tensor]
        ) -> float:
            """
            计算query embedding和domain embedding的余弦相似度
            
            Args:
                query_embedding: query的embedding
                domain_embedding: domain的embedding
                
            Returns:
                相似度分数 (0-1)
            """
            # 计算余弦相似度
            similarity: list = [torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                domain_embedding.unsqueeze(0)
            ).item() for domain_embedding in domain_embedding_list]
            
            return sum(similarity)


        
        for domain_name, domain_embedding_list in self.slm_embeddings.items():
            similarity = calculate_similarity_v2(query_embedding, domain_embedding_list)
            similarities[domain_name] = similarity

            print(f'{query=}, {domain_name=}, {similarity=}')

        # 找出最高相似度
        if not similarities:
            return None, 0.0, {}

        similarities = self.remove_unrelated(similarities)
        similarities = self.normalized(similarities)
        for domain_name, similarity in similarities.items():
            print(f"  - {domain_name}: {similarity:.4f}")

        # Find best domain expert regarding similarity
        best_domain, best_similarity = max(similarities.items(), key=lambda x: x[1])

        assert best_domain is not None
        best_similarity = similarities[best_domain]
        
        # 判断是否超过阈值
        if best_similarity < self.similarity_threshold:
            print(f"\n[Selection] Best similarity ({best_similarity:.4f}) below threshold ({self.similarity_threshold})")
            print(f"[Selection] Will use Web Search instead")
            return None, best_similarity, similarities
        
        print(f"\n[Selection] Selected domain: {best_domain} (similarity: {best_similarity:.4f})")
        return best_domain, best_similarity, similarities

    
    def generate_background_info(
        self,
        domain_name: str,
        query: str,
        max_length: int = 300
    ) -> str:
        """
        使用选定的SLM生成背景信息
        
        Args:
            domain_name: 选中的领域名称
            query: 原始查询
            max_length: 生成的最大长度
            
        Returns:
            生成的背景信息
        """
        print(f"\n[Step 4] Generating background information using {domain_name} SLM...")
        
        # 加载模型
        slm = self.load_slm(domain_name)
        model = slm["model"]
        tokenizer = slm["tokenizer"]
        
        # 构造prompt
        prompt = f"""Based on the following query, provide relevant background information and context to help understand the topic better. Do NOT answer the question directly.

Query: {query}

Background Information:"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的背景信息（去掉prompt部分）
        background_info = generated_text[len(prompt):].strip()
        
        print(f"Background info generated ({len(background_info)} chars)")
        return background_info
    
    def web_search_fallback(
        self,
        query: str,
        model_name: str = "qwen-plus",
        max_length: int = 300
    ) -> str:
        """
        使用Web搜索获取背景信息（当没有匹配的SLM时）
        
        Args:
            query: 原始查询
            model_name: 使用的模型名称
            max_length: 最大长度
            
        Returns:
            搜索得到的背景信息
        """
        print(f"\n[Step 5] Using Web Search to retrieve information...")
        
        result = self.Qianwen_search_result(
            query=query,
            model_name=model_name,
            Web_search=True,
            max_length=max_length
        )
        
        return result
    
    @staticmethod
    def Qianwen_search_result(query, model_name, Web_search=True, entity_search=None, max_length=300):
        """
        Perform web search using the Qianwen API to get relevant information for a query.
        Optimized for OpenAI version 0.27.6.
        """
        if entity_search is None:
            entity_search = []
        
        try:
            # Set up for OpenAI version 0.27.6
            import openai
            
            # Configure API credentials
            api_key = os.getenv('QIANWEN_API_KEY', os.getenv("DASHSCOPE_API_KEY", "xxx"))
            openai.api_key = api_key
            openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
            # Create messages for the API
            messages = [
                {'role': 'system', 'content': "You are a research assistant. ONLY provide relevant background information, concepts, and definitions to understand the topic. Do NOT solve the problem or give direct answers. Always respond in English."},
                {'role': 'user', 'content': f"Find ONLY background information in English about this topic (DO NOT solve it): {query}"}
            ]
            
            # Make the API call using the 0.27.6 format
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_length,
                # For 0.27.6, use custom parameters
                enable_search=Web_search,
                search_options={
                    "forced_search": True
                }
            )
            
            # Extract the assistant's response for 0.27.6 format
            if 'choices' in completion and len(completion['choices']) > 0:
                search_result = completion['choices'][0]['message']['content']
                return f"{search_result}"
            
            # Fallback if the expected response format is different
            return f"{str(completion)}"
                
        except Exception as e:
            print(f"Web search error: {e}")
            # Provide more detailed error handling
            import traceback
            traceback.print_exc()
            return ""  # Return empty string on error
    
    def process_query(
        self,
        query: str,
        max_background_length: int = 300
    ) -> Dict[str, any]:
        """
        完整的查询处理流程
        
        Args:
            query: 用户输入的查询
            max_background_length: 背景信息的最大长度
            
        Returns:
            包含处理结果的字典
        """
        print("="*80)
        print(f"Processing Query: {query}")
        print("="*80)
        
        # 选择最佳SLM
        selected_domain, similarity, all_similarities = self.select_best_slm_v2(query)
        method = None
        result = {}

        if self.is_llm_involved:
            # 生成背景信息
            if selected_domain is not None:
                # 使用选中的SLM生成背景信息
                background_info = self.generate_background_info(
                    selected_domain,
                    query,
                    max_background_length
                )
                method = "domain_slm"
            else:
                # 使用Web搜索
                background_info = self.web_search_fallback(
                    query,
                    max_length=max_background_length
                )
                method = "web_search"

            # 组合增强后的query
            enhanced_query = f"""Original Query: {query}

    Background Information:
    {background_info}

    Please answer the original query considering the background information provided above."""

            # 返回结果
            result = {
                "original_query": query,
                "selected_domain": selected_domain,
                "similarity_score": similarity,
                "all_similarities": all_similarities,
                "method_used": method,
                "background_info": background_info,
                "enhanced_query": enhanced_query
            }

        print("\n" + "="*80)
        print("Processing Complete!")
        print(f"Method: {'no llm' if not self.is_llm_involved else method}")
        if selected_domain:
            print(f"Domain: {selected_domain}")
        print("="*80 + "\n")
        
        return result
    
    def unload_slm(self, domain_name: str):
        """
        卸载指定的SLM以释放内存
        """
        if domain_name in self.loaded_models:
            self.unload_model(self.loaded_models[domain_name])
            del self.loaded_models[domain_name]
            print(f"SLM for '{domain_name}' unloaded")

    def unload_model(self, loaded_model):
        """Properly unload model and free memory"""
        import gc
        import torch

        model = loaded_model['model']

        # Delete model
        if model is not None:
            try:
                model.cpu()
            except:
                pass

        # Clear cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
    
    def unload_all_slms(self):
        """
        卸载所有已加载的SLM
        """
        if not self.loaded_models:
            return 
        
        for domain_name in list(self.loaded_models.keys()):
            self.unload_slm(domain_name)
        print("All SLMs unloaded")


def main():
    """
    示例使用
    """
    # 初始化路由器
    router = SmartQueryRouter(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.6
    )
    
    # 注册4个领域专家SLM
    # 注意：您需要替换为实际的模型ID和LoRA权重路径
    
    # 1. 医疗领域
    router.register_slm_v2(
        domain_name="medical",
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct",  # 替换为您的模型ID
        lora_weights_path = "Arthur-77/QWEN2.5-1.5B-medical-finetuned",  # 替换为您的LoRA权重路径
        domain_description_list=['possesses deep knowledge of diseases, treatments, medications, medical procedures, anatomy, physiology, and clinical practices',
                                 'capable of analyzing complex patient presentations and medical histories to identify likely conditions',
                                   'understands the mechanisms, interactions, and evidence-based applications of pharmaceuticals and therapies',
                                     'expert in interpreting diagnostic results, from lab values to medical imaging',
                                     'proficient in following and explaining clinical guidelines, best practices, and standard-of-care protocols',
                                       'maintains a current, evidence-based understanding of the latest medical research and advancements',
                                         'skilled in risk assessment, differential diagnosis, and tailoring treatment plans to individual patient factors'
                                         ]
    )
    
    # 2. 金融领域
    router.register_slm_v2(
        domain_name="finance",
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_weights_path = "WiroAI/WiroAI-Finance-Qwen-1.5B",
        domain_description_list=['possesses comprehensive knowledge of stock markets, investments, banking, financial analysis, cryptocurrency, trading, and economic theories',
'capable of analyzing market trends, financial statements, and macroeconomic indicators',
'understands the principles, risks, and strategies behind various asset classes and investment vehicles',
'expert in interpreting financial data, valuation models, and performance metrics',
'proficient in applying financial regulations, compliance standards, and trading protocols',
'maintains a current understanding of global economic events and their market implications',
'skilled in portfolio construction, risk management, and financial forecasting']
    )
    
    # 3. Math
    router.register_slm_v2(
        domain_name="math",
        base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        lora_weights_path="Erlanggaa/qwen1.5b",
        domain_description_list=['possesses comprehensive knowledge of algebra, calculus, geometry, statistics, mathematical reasoning, problem solving, equations, and quantitative analysis',
 'capable of analyzing complex mathematical structures, proofs, and real-world problem scenarios',
 'understands the foundational principles, theorems, and applications across diverse mathematical branches',
 'expert in constructing formal proofs, interpreting statistical models, and solving intricate equations',
 'proficient in applying logical reasoning, algorithmic thinking, and rigorous analytical methodologies',
 'maintains a deep understanding of both pure theoretical concepts and practical computational techniques',
 'skilled in abstract reasoning, quantitative modeling, and translating problems into mathematical frameworks']
    )
    
    # 4. 技术领域
    router.register_slm_v2(
        domain_name="technology",
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct",
        lora_weights_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        domain_description_list=['possesses deep expertise in machine learning algorithms, neural network architectures, deep learning frameworks, and AI model development',
 'capable of designing, training, and deploying models for supervised, unsupervised, and reinforcement learning tasks',
 'understands the mathematical foundations including linear algebra, calculus, probability theory, and statistical inference underlying ML systems',
 'expert in feature engineering, model selection, hyperparameter tuning, and addressing overfitting, underfitting, and bias-variance tradeoffs',
 'proficient in modern ML frameworks and libraries such as TensorFlow, PyTorch, scikit-learn, and tools for distributed training and model serving',
 'maintains current knowledge of state-of-the-art research, including transformer architectures, diffusion models, large language models, and multimodal AI systems',
 'skilled in MLOps practices, experiment tracking, model evaluation metrics, data pipeline construction, and production deployment of AI systems',
 'understands ethical considerations, fairness, interpretability, and responsible AI practices in model development and deployment']
    )
    
    # 测试不同类型的查询
    test_queries = [
        "What are the symptoms and treatment options for type 2 diabetes?",
        "How does the Federal Reserve's interest rate decision affect the stock market?",
        "Explain the relationship between the Fundamental Theorem of Calculus and the concept of antiderivatives",
        "Explain the difference between supervised and unsupervised learning in machine learning",
        "How to clean toilet?"  # 不属于任何领域，应该使用Web搜索
    ]

    for query in test_queries:
        result = router.process_query(query)
        
        # 打印结果摘要
        print(f"\nQuery: {query}")
        print(f"Selected Domain: {result['selected_domain']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Method Used: {result['method_used']}")
        print(f"Background Info Preview: {result['background_info'][:200]}...")
        print("\n" + "-"*80 + "\n")
    
    # 清理
    router.unload_all_slms()


if __name__ == "__main__":
    main()
