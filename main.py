from smart_query_router import SmartQueryRouter
import config
from factuality import FactualChecker
from chat import chat
import os
from pathlib import Path
import json
import shutil


_PREVIEW_LEN = 100
MODEL = 'Qwen-plus'

def main(queries):
    router = SmartQueryRouter(
        embedding_model_name=config.EMBEDDING_MODEL,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        device=config.DEVICE,
        is_llm_involved=True   # no llm involved means no any genearation, only routing
    )

    checker = FactualChecker()
    
    for domain_name, domain_config in config.SLM_DOMAINS.items():
        router.register_slm_v2(
            domain_name=domain_name,
            base_model_id=domain_config["base_model_id"],
            lora_weights_path=domain_config["lora_weights_path"],
            domain_description_list=domain_config["domain_description_list"]
        )

    conversation = []
    
    for i, query in enumerate(queries, 1):
        result = router.process_query(
            query=query,
            max_background_length=config.MAX_BACKGROUND_LENGTH
        )

        # no llm involved => no generation
        if not result:
            return None

        answer: str = chat(query)

        print(f'Query {i}: {query[:_PREVIEW_LEN]}{'...' if len(query) >= _PREVIEW_LEN else ''}')
        print(f"Selected Domain: {result['selected_domain'] or 'None (Web Search)'}", end='\n\n')
        
        print("[Similarity Scores for All Domains]")
        for domain, score in result['all_similarities'].items():
            print(f"  {domain}: {score:.4f}")
        
        print(f'Preview: {result['background_info'][:_PREVIEW_LEN]}{'...' if len(result['background_info']) >= _PREVIEW_LEN else ''}')
        print('[Factuality]: Checking if expanded query is valid')
        
        is_valid = checker.factuality_check(result['enhanced_query'])
        print(f'The Expanded Query is {"Valid" if is_valid else "Invalid"}')

        new_query = result['enhanced_query'] if is_valid else query
        answer_enhanced: str = chat(new_query)

        print(f'[Answer]\n{answer[:2*_PREVIEW_LEN]}{'...' if len(answer) >= 2*_PREVIEW_LEN else ''}', end='\n\n')

        conversation.append((query, answer, answer_enhanced))
        
    return conversation
    
if __name__ == "__main__":
    queries = [
        # "What are the symptoms and treatment options for type 2 diabetes?", # medical
        # "How does the Federal Reserve's interest rate decision affect the stock market?", # finance
        # "Explain the relationship between the Fundamental Theorem of Calculus and the concept of antiderivatives", # math
        # "Explain the difference between supervised and unsupervised learning in machine learning", # technology
        "How to clean toilet?",  # 不属于任何领域，应该使用Web搜索
        # "Best recipe for chocolate cake", # web search
        # "what is sparse attention in machine learning", # technology
    ]

    conversations = main(queries)

    if conversations:
        os.makedirs('output/archived', exist_ok=True)
        result_path = Path('output/result.txt')

        if result_path.exists():
            file_index = [int(str(Path(file_name).stem).split('_')[1]) for file_name in os.listdir('output/archived')] or [0]
            next_index = max(file_index) + 1 # extract index
            archived_path = result_path.parent / 'archived' / ('result_' + str(next_index) + result_path.suffix)
            shutil.move(result_path, archived_path)

        with open(result_path, 'a') as output:
            for query, answer, answer_enhanced in conversations:
                out_format = {'query': query, 'answer': answer, 'answer_enhanced': answer_enhanced}
                json.dump(out_format, fp=output)
    else:
        print('Done')
