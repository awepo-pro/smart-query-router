"""
配置文件示例 - Smart Query Router Configuration
"""

# ============================================================================
# 基础配置
# ============================================================================

# Embedding模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 其他可选的embedding模型:
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 支持多语言
# - "sentence-transformers/all-mpnet-base-v2"  # 更高质量
# - "BAAI/bge-small-en-v1.5"  # 中文友好

# 相似度阈值 (0-1)
# 低于此阈值将使用Web搜索而不是SLM
SIMILARITY_THRESHOLD = 0.5

# 计算设备
DEVICE = None  # "cpu" (default), or 'cuda'

# ============================================================================
# SLM领域配置
# ============================================================================

# SLM_DOMAINS = {
#     "medical": {
#         "base_model_id": "meta-llama/Llama-2-7b-hf",
#         "lora_weights_path": "./lora_weights/medical_lora",
#         "domain_description": (
#             "Medical and healthcare domain, including diseases, treatments, "
#             "medications, medical procedures, anatomy, physiology, diagnostics, "
#             "patient care, and clinical practices"
#         )
#     },
    
#     "finance": {
#         "base_model_id": "meta-llama/Llama-2-7b-hf",
#         "lora_weights_path": "./lora_weights/finance_lora",
#         "domain_description": (
#             "Finance and economics domain, including stock market, investments, "
#             "banking, financial analysis, cryptocurrency, trading, risk management, "
#             "portfolio optimization, and economic theories"
#         )
#     },
    
#     "legal": {
#         "base_model_id": "meta-llama/Llama-2-7b-hf",
#         "lora_weights_path": "./lora_weights/legal_lora",
#         "domain_description": (
#             "Legal domain, including laws, regulations, contracts, court cases, "
#             "legal procedures, constitutional law, intellectual property, "
#             "litigation, and legal rights"
#         )
#     },
    
#     "technology": {
#         "base_model_id": "meta-llama/Llama-2-7b-hf",
#         "lora_weights_path": "./lora_weights/tech_lora",
#         "domain_description": (
#             "Technology domain, including programming, software development, "
#             "artificial intelligence, machine learning, deep learning, computer science, "
#             "algorithms, data structures, system design, and technical systems"
#         )
#     }
# }

SLM_DOMAINS = {
    "medical": {
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_weights_path": "models--Arthur-77--QWEN2.5-1.5B-medical-finetuned/snapshots/3ff4b05835d9d0af645f2a4e9c33edeb149bcef8/",
        "domain_description_list": ['possesses deep knowledge of diseases, treatments, medications, medical procedures, anatomy, physiology, and clinical practices',
                                'capable of analyzing complex patient presentations and medical histories to identify likely conditions',
                                'understands the mechanisms, interactions, and evidence-based applications of pharmaceuticals and therapies',
                                'expert in interpreting diagnostic results, from lab values to medical imaging',
                                'proficient in following and explaining clinical guidelines, best practices, and standard-of-care protocols',
                                'maintains a current, evidence-based understanding of the latest medical research and advancements',
                                'skilled in risk assessment, differential diagnosis, and tailoring treatment plans to individual patient factors']
    },
    
    "finance": {
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_weights_path": "models--DreamGallery--Qwen-Qwen2.5-1.5B-1727452785/snapshots/6c4a0b3f7975eed5f016ea7842bfcca6d540d0e8/",
        "domain_description_list": ['possesses comprehensive knowledge of stock markets, investments, banking, financial analysis, cryptocurrency, trading, and economic theories',
                                'capable of analyzing market trends, financial statements, and macroeconomic indicators',
                                'understands the principles, risks, and strategies behind various asset classes and investment vehicles',
                                'expert in interpreting financial data, valuation models, and performance metrics',
                                'proficient in applying financial regulations, compliance standards, and trading protocols',
                                'maintains a current understanding of global economic events and their market implications',
                                'skilled in portfolio construction, risk management, and financial forecasting']
    },
    
    "math": {
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_weights_path": "models--SriSanthM--Qwen-1.5B-Tweet-Generations/snapshots/036669d2128f332f5417e2d11de99bfd058f5f64",
        "domain_description_list": ['possesses comprehensive knowledge of algebra, calculus, geometry, statistics, mathematical reasoning, problem solving, equations, and quantitative analysis',
                                'capable of analyzing complex mathematical structures, proofs, and real-world problem scenarios',
                                'understands the foundational principles, theorems, and applications across diverse mathematical branches',
                                'expert in constructing formal proofs, interpreting statistical models, and solving intricate equations',
                                'proficient in applying logical reasoning, algorithmic thinking, and rigorous analytical methodologies',
                                'maintains a deep understanding of both pure theoretical concepts and practical computational techniques',
                                'skilled in abstract reasoning, quantitative modeling, and translating problems into mathematical frameworks']
    },
    
    "technology": {
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "lora_weights_path": "models--silent666--Qwen-Qwen2.5-1.5B-Instruct-1727478552/snapshots/1c048b9dacacbf457396fcefe1c136abc053c069/",
        "domain_description_list": ['possesses deep expertise in machine learning algorithms, neural network architectures, deep learning frameworks, and AI model development',
                                'capable of designing, training, and deploying models for supervised, unsupervised, and reinforcement learning tasks',
                                'understands the mathematical foundations including linear algebra, calculus, probability theory, and statistical inference underlying ML systems',
                                'expert in feature engineering, model selection, hyperparameter tuning, and addressing overfitting, underfitting, and bias-variance tradeoffs',
                                'proficient in modern ML frameworks and libraries such as TensorFlow, PyTorch, scikit-learn, and tools for distributed training and model serving',
                                'maintains current knowledge of state-of-the-art research, including transformer architectures, diffusion models, large language models, and multimodal AI systems',
                                'skilled in MLOps practices, experiment tracking, model evaluation metrics, data pipeline construction, and production deployment of AI systems',
                                'understands ethical considerations, fairness, interpretability, and responsible AI practices in model development and deployment']
    }
}

# ============================================================================
# Web搜索配置 (Qianwen API)
# ============================================================================

# Qianwen API配置
QIANWEN_API_KEY = "xxx"  # 从环境变量或这里读取
QIANWEN_MODEL = "qwen-plus"  # 可选: "qwen-plus", "qwen-turbo", "qwen-max"

# Web搜索参数
WEB_SEARCH_MAX_LENGTH = 300

# ============================================================================
# 生成配置
# ============================================================================

# 背景信息生成的最大长度
MAX_BACKGROUND_LENGTH = 300

# 生成参数
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 300
}

# ============================================================================
# 日志配置
# ============================================================================

# 是否打印详细日志
VERBOSE = True

# 日志级别
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
