
import json
from typing import Any
import torch as th
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.utils import LinearSchedule

from ddo.services.utils_data import get_disease_index_dict, get_symptom_index_dict
from ddo.paths import get_policy_run_dir
from ddo.services.policy import SymptomInquiryActorCriticPolicy
from ddo.state_model.state import ConsultationState

class DummyEvalEnvs:
    def __init__(self):
        self.envs = [object()]

def linear_schedule(initial_value: float):
    return LinearSchedule(
        start=initial_value,
        end=initial_value * 0.1,
        end_fraction=1,
    )

class PolicyCandidateSampler:
    def __init__(
        self,
        *,
        dataset_name: str,
        device: str = "cpu",
        window_size: int | None = None,
        num_samples: int | None = None,
        importance_threshold: float | None = None,
    ):
        self.dataset_name = dataset_name
        self.device = th.device(device)

        policy_run_dir = get_policy_run_dir(dataset_name)
        settings_path = policy_run_dir / "best_settings.json"
        policy_path = policy_run_dir / "policy.pth"

        with open(settings_path, "r", encoding="utf-8") as f:
            self.settings = json.load(f)
        
        self.disease_index_dict = get_disease_index_dict(dataset_name)
        self.symptom_index_dict = get_symptom_index_dict(dataset_name)

        self.index_to_disease = {
            idx: disease for disease, idx in self.disease_index_dict.items()
        }
        self.index_to_symptom = {
            idx: symptom for symptom, idx in self.symptom_index_dict.items()
        }

        self.disease_num = len(self.disease_index_dict)
        self.symptom_num = len(self.symptom_index_dict)

        self.window_size = window_size or self.settings.get("window_size", 3)
        self.num_samples = num_samples or self.settings.get("num_samples", 6)

        # 重要性阈值，低于该阈值的症状将被过滤掉
        self.importance_threshold = (
            importance_threshold
            if importance_threshold is not None
            else self.settings.get("importance_threshold", 0.0)
        )

        observation_space = spaces.Box(
            low=np.array([-1] * self.symptom_num + [0] * self.disease_num),
            high=np.array([1] * self.symptom_num + [1] * self.disease_num),
            shape=(self.symptom_num + self.disease_num,),
            dtype=np.float32,
        )

        action_space = spaces.Discrete(self.symptom_num + 1)

        policy_kwargs = {
            "net_arch": {
                "pi": self.settings["net_arch_pi"],
                "vf": self.settings["net_arch_vf"],
            },
            "activation_fn": th.nn.ReLU,
            "dataset_name": dataset_name,
            "importance_threshold": self.importance_threshold,
            "window_size": self.window_size,
            "num_samples": self.num_samples,

            # 构造函数要求的参数
            "retry":self.settings.get("retry", 1),
            "llm_name":self.settings.get("llm_name", "none"),
            "llm":None,
            "tokenizer":None,
            "seed":self.settings.get("seed", 42),
            "eval_envs":DummyEvalEnvs(),
        }

        lr_schedule = linear_schedule(self.settings.get("learning_rate", 5e-5))

        self.policy = SymptomInquiryActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **policy_kwargs,
        )

        state_dict = th.load(policy_path, map_location=self.device)
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.to(self.device)
        self.policy.eval()

    def build_observation(self, state: ConsultationState) -> th.Tensor:
        obs = np.zeros(self.symptom_num + self.disease_num, dtype=np.float32)

        positive_symptoms = state.diagnosis_state.positive_symptoms
        negative_symptoms = state.diagnosis_state.negative_symptoms

        for symptom in positive_symptoms:
            if symptom in self.symptom_index_dict:
                idx = self.symptom_index_dict[symptom]
                obs[idx] = 1.0

        for symptom in negative_symptoms:
            if symptom in self.symptom_index_dict:
                idx = self.symptom_index_dict[symptom]
                obs[idx] = -1.0

        raw_confidence = state.diagnosis_state.raw_diagnosis_confidence
        # normalized_confidence = state["diagnosis_state"].get("diagnosis_confidence", {})

        disease_confidence = raw_confidence

        offset = self.symptom_num
        for disease, idx in self.disease_index_dict.items():
            obs[offset + idx] = disease_confidence.get(disease, 0.0)

        return th.from_numpy(obs).float().unsqueeze(0).to(self.device)

    def get_top_disease_from_obs(self, obs: th.Tensor) -> list[str]:
        disease_part = obs[0, self.symptom_num:self.symptom_num + self.disease_num]
        top_indices = th.argsort(disease_part, descending=True)[:self.window_size]

        return [
            self.index_to_disease[int(idx.item())]
            for idx in top_indices
        ]
    
    def sample_candidate_symptoms(
            self,
            state: ConsultationState,
            *,
            extra_masking_action_ids: list[int] | None = None,
        ) -> dict[str, Any]:

        obs = self.build_observation(state)

        # 加载被 Inquiry Agent 拒绝过的症状动作列表，并进行清洗和去重

        # retry 时额外 mask 掉之前被 Inquiry Agent 拒绝过的症状动作。
        # {} 表示集合推导式，会自动去重
        # sorted 排序后会把集合变成列表
        cleaned_extra_masking_action_ids = sorted({
            int(action_id)
            for action_id in (extra_masking_action_ids or [])
            if 0 <= int(action_id) < self.symptom_num
        })

        # 变张量
        extra_masking_tensor = None
        if cleaned_extra_masking_action_ids:
            extra_masking_tensor = th.tensor(
                cleaned_extra_masking_action_ids,
                dtype=th.long,
                device=self.device,
            )

        with th.no_grad():
            self.policy.stage = "test"

            latent_pi = self.policy.mlp_extractor.forward_actor(obs)
            distribution = self.policy._get_action_dist_from_latent_with_mask(
                latent_pi = latent_pi,
                features = obs,
                extra_masking = extra_masking_tensor,
                extra_masking_env_id = 0,           
            )

            sampled_actions = [
                int(distribution.sample().item())
                for _ in range(self.num_samples)
            ]

        # 从采样的动作中提取候选症状和是否包含终止动作，去重了
        candidate_symptoms = []
        has_termination_action = False

        for action_id in sampled_actions:
            if action_id == self.symptom_num:
                has_termination_action = True
                continue

            symptom = self.index_to_symptom.get(action_id)

            if symptom is not None and symptom not in candidate_symptoms:
                candidate_symptoms.append(symptom)

        # 获取诊断置信度排名前 window_size 的疾病列表
        # top_disease = self.get_top_disease_from_obs(obs)

        return {
            "candidate_actions_ids": sampled_actions,
            "candidate_symptoms": candidate_symptoms,
            "has_termination_action": has_termination_action,
            # "top_disease_by_policy_obs": top_disease,

            "extra_masking_action_ids": cleaned_extra_masking_action_ids,
            "num_samples": self.num_samples,
            "window_size": self.window_size,
            "importance_threshold": self.importance_threshold,
        }
