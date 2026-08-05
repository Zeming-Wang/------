from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetContext:
    dataset_name: str
    candidate_diseases: tuple[str, ...]
    disease_knowledge: dict
    disease_index_dict: dict[str, int]
    symptom_index_dict: dict[str, int]