from smart_query_router import SmartQueryRouter
import config
from factuality import FactualChecker
from chat import chat


_PREVIEW_LEN = 100
MODEL = 'Qwen-plus'

def main(queries):
    router = SmartQueryRouter(
        embedding_model_name=config.EMBEDDING_MODEL,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        device=config.DEVICE
    )

    checker = FactualChecker()
    
    for domain_name, domain_config in config.SLM_DOMAINS.items():
        router.register_slm(
            domain_name=domain_name,
            base_model_id=domain_config["base_model_id"],
            lora_weights_path=domain_config["lora_weights_path"],
            domain_description=domain_config["domain_description"]
        )

    answers = []
    
    for i, query in enumerate(queries, 1):
        result = router.process_query(
            query=query,
            max_background_length=config.MAX_BACKGROUND_LENGTH
        )

        print(f'Query {i}: {query[:_PREVIEW_LEN]}{'...' if len(query) >= _PREVIEW_LEN else ''}')
        print(f"Selected Domain: {result['selected_domain'] or 'None (Web Search)'}")
        print(f"Similarity Score: {result['similarity_score']:.4f}", end='\n\n')
        
        print("[Similarity Scores for All Domains]")
        for domain, score in result['all_similarities'].items():
            print(f"  {domain}: {score:.4f}")
        
        print(f'Preview: {result['background_info'][:_PREVIEW_LEN]}{'...' if len(result['background_info']) >= _PREVIEW_LEN else ''}')
        print('[Factuality]: checking if expanded query is valid')
        
        is_valid = checker.factuality_check(result['enhanced_query'])
        print(f'The Expanded Query is: {is_valid}')

        new_query = result['enhanced_query'] if is_valid else query
        answer: str = chat(new_query)

        print(f'[Answer]\n{answer[:2*_PREVIEW_LEN]}{'...' if len(answer) >= 2*_PREVIEW_LEN else ''}', end='\n\n')

        answers.append(answer)
        
    return answers
        
if __name__ == "__main__":
    queries = [
        "What are the symptoms of COVID-19?",  # 医疗领域
        "How to invest in the stock market?",  # 金融领域
        "What is a patent?",  # 法律领域
        "Explain neural networks",  # 技术领域
        "Best recipe for chocolate cake"  # 不属于任何领域 -> Web搜索
    ]

    answers = main(queries)
