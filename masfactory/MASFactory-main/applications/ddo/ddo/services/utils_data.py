import json
import os

from ddo.paths import DATA_DIR

def load_dataset(dataset_name: str, stage: str = "test"):
    file_path = os.path.join(DATA_DIR, dataset_name, f"{stage}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def get_disease_knowledge(dataset_name: str, candidate_diseases: list):
    disease_knowledge = {
        candidate_disease: {} for candidate_disease in candidate_diseases
    }
    
    file_path = os.path.join(DATA_DIR, dataset_name, "empirical_knowledge.json")
    with open(file_path, "r", encoding="utf-8") as f:
        empirical_knowledge = json.load(f)
        for candidate_disease in candidate_diseases:
            disease_knowledge[candidate_disease]["empirical_knowledge"] = empirical_knowledge[candidate_disease]

    """
    结构如下：
    {
        "感冒": {
            "empirical_knowledge": {
                "咳嗽": 0.85,
                "发热": 0.70,
                "流涕": 0.90
            }
        },
        "肺炎": {
            "empirical_knowledge": {
                "咳嗽": 0.95,
                "发热": 0.88,
                "呼吸困难": 0.60
            }
        }
    }
    """

    return disease_knowledge

def get_disease_index_dict(dataset_name: str):
    disease_index_dict = {}

    file_path = os.path.join(DATA_DIR, dataset_name, "disease_corpurs.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():
            disease_index_dict[line.strip()] = len(disease_index_dict)
    return disease_index_dict

def get_symptom_index_dict(dataset_name: str):
    symptom_index_dict = {}

    file_path = os.path.join(DATA_DIR, dataset_name, "symptom_corpurs.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():
            symptom_index_dict[line.strip()] = len(symptom_index_dict)
    return symptom_index_dict
