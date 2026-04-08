import sys
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

import json
from typing import List, Union

def extract_texts_for_pipeline(filepaths: Union[str, List[str]], include_title: bool = True) -> List[str]:
    if isinstance(filepaths, str):
        filepaths =[filepaths]
        
    extracted_texts =[]
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                for article in data:
                    text_parts =[]
                    
                    if include_title and article.get("title"):
                        text_parts.append(f"Title: {article['title']}")
                        
                    if article.get("abstract"):
                        text_parts.append(article["abstract"])
                        
                    combined_text = "\n\n".join(text_parts)
                    
                    if combined_text.strip():
                        extracted_texts.append(combined_text)
                        
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file '{filepath}' is not a valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred reading '{filepath}': {e}")
            
    return extracted_texts

# if __name__ == "__main__":
#     texts = extract_texts_for_pipeline("rag_documents/general_case_reports.json", include_title=True)
#     print(f"Loaded {len(texts)} texts.")
    
#     if texts:
#         print("\n--- Preview of Document 1 ---")
#         print(texts[0][:300] + "...\n")
        
#     target_entities = [
#         "PATIENT DEMOGRAPHIC",
#         "DISEASE OR CONDITION",
#         "SYMPTOM OR CLINICAL_FINDING",
#         "DRUG OR MEDICATION",
#         "PROCEDURE OR SURGERY",
#         "DIAGNOSTIC TEST OR DEVICE",
#         "ANATOMY",
#         "GENETIC MUTATION OR MARKER"
#     ]
    
#     relation_labels = [
#         "TREATS OR MANAGES",
#         "CAUSES OR ADVERSE EFFECT OF",
#         "PRESENTS WITH",
#         "DIAGNOSED BY",
#         "MIMICS OR DIFFERENTIAL DIAGNOSIS",
#         "LOCATED IN OR AFFECTS",
#         "ASSOCIATED WITH GENETIC MARKER",
#         "COMPLICATES OR CO-OCCURS WITH"
#     ]

    
#     builder = retrico.RetriCoBuilder(name="medical_pipeline")
#     builder.chunker(method="sentence")
#     builder.relex_gliner(model="knowledgator/gliner-relex-large-v1.0", entity_labels=target_entities, relation_labels=relation_labels)
#     builder.graph_writer()
#     builder.graph_writer(json_output="data/graph_data/pretrained_general.json")
#     executor = builder.build(verbose=True)
#     result = executor.run(texts=texts)
#     stats = result.get("writer_result")
#     print(f"Entities: {stats['entity_count']}, Relations: {stats['relation_count']}")
