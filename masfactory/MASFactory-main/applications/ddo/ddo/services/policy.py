from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
from gymnasium import spaces
from torch import nn
from typing import Callable, Optional, Tuple
from ddo.services.utils_data import get_disease_knowledge, get_disease_index_dict, get_symptom_index_dict

class ACNet(nn.Module):
    def __init__(
        self,
        features_dim: int,
        net_arch
    ):
        super().__init__()
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, net_arch['pi'][0]), nn.ReLU(),
            nn.Linear(net_arch['pi'][0], net_arch['pi'][1]), nn.ReLU(),
            nn.Linear(net_arch['pi'][1], net_arch['pi'][2]), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, net_arch['vf'][0]),  nn.ReLU(),
        )
        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class SymptomInquiryActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        dataset_name: str,
        importance_threshold: float,
        window_size: int,
        num_samples: int,
        retry: int,
        eval_envs: DummyVecEnv,
        llm_name: str,
        llm,
        tokenizer,
        seed,
        *args,
        **kwargs,  # net_arch, activation_fn
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.features_dim = observation_space.shape[0]
        self.net_arch = kwargs['net_arch'] 
        self.activation_fn = kwargs['activation_fn'], 
        self.disease_index_dict = get_disease_index_dict(dataset_name) 
        self.symptom_index_dict = get_symptom_index_dict(dataset_name) 
        self.disease_index_to_name = list(self.disease_index_dict.keys())
        self.symptom_index_to_name = list(self.symptom_index_dict.keys())
        self.disease_num = len(self.disease_index_dict.keys()) 
        self.symptom_num = len(self.symptom_index_dict.keys()) 
        self.disease_knowledge = get_disease_knowledge(dataset_name, list(self.disease_index_dict.keys())) 
        self.importance_threshold = importance_threshold
        self.window_size = window_size 
        self.num_samples = num_samples 
        self.retry = retry 
        self.eval_envs = eval_envs 
        self.decision_info = [{"retry_history": [], "candidate_symptoms": None, "selected_symptom": None, "selection_reasoning": None}] * len(eval_envs.envs)
        self.llm_name = llm_name 
        self.llm = llm 
        self.tokenizer = tokenizer 
        self.stage = "train"

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ACNet(
            self.features_dim,
            self.net_arch,
        )

    def _get_action_dist_from_latent_with_mask(self, latent_pi: th.Tensor, features: th.Tensor, extra_masking: th.Tensor = None, extra_masking_env_id: int = 0) -> Distribution:
        action_logits = self.action_net(latent_pi)
        # print(action_logits)
        mask = self.get_current_masking(features)

        if self.stage != "train" and extra_masking is not None:
            # extra_masking_env_id是一个整数，表示在哪个环境中需要额外屏蔽，范围是0到env_size-1
            # extra_masking是一个布尔张量，类似列表，长度等于动作空间大小，表示哪些动作需要额外屏蔽
            # symptom_num是症状的数量，动作空间的最后一个动作是终止动作，所以索引为symptom_num，即允许选择终止动作
            mask[extra_masking_env_id, extra_masking] = False
            mask[extra_masking_env_id, self.symptom_num] = True
        mask_logits = th.zeros_like(action_logits).masked_fill_(~mask, float('-inf'))
        action_logits = action_logits + mask_logits
        
        
        if isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent_with_mask(latent_pi, features)

    def get_current_masking(self, features):

        # features 就是obs
        env_size = len(features)
        mask = th.zeros([env_size, self.symptom_num + 1], device=self.device)
        mask[:, self.symptom_num] = 1

        # obs的前面部分是症状的状态，后面部分是疾病的置信度
        features_symptom_status = features[:, :self.symptom_num]
        features_diagnostic_confidence = features[:, self.symptom_num:self.symptom_num + self.disease_num]

        # 根据当前的诊断置信度，选取top-k疾病，并将这些疾病相关的症状加入mask中
        top_diseases_indices = th.argsort(features_diagnostic_confidence, dim=1, descending=True)[:, :self.window_size]

        # 构建疾病到症状的映射，得到每个疾病相关的症状索引列表，是数而不是str
        # [[3,1,...],[],...]，每个子列表是一个疾病相关的症状索引列表
        disease_to_symptoms = [
            [self.symptom_index_dict[symptom] for symptom in self.disease_knowledge[disease]["empirical_knowledge"].keys()]
            for disease in self.disease_index_dict.keys()
        ]
        # 张量化
        disease_to_symptoms = [th.tensor(indices, device=self.device) for indices in disease_to_symptoms]


        for i in range(env_size):
            current_top_diseases = top_diseases_indices[i]

            # 拼接当前top-k疾病相关的症状索引，去重后得到当前需要关注的症状索引列表
            related_symptoms = th.cat([disease_to_symptoms[idx] for idx in current_top_diseases])
            unique_symptoms = th.unique(related_symptoms)

            # 构建mask
            mask[i, unique_symptoms] = (features_symptom_status[i, unique_symptoms] == 0).float()
            
            # 如果不是训练阶段，进一步过滤掉那些与当前top-k疾病相关但重要性较低的症状
            if self.stage != "train":
                for symptom_idx in unique_symptoms:
                    symptom_name = self.symptom_index_to_name[symptom_idx.item()]
                    is_unimportant = all(
                        self.disease_knowledge[self.disease_index_to_name[disease_idx]]["empirical_knowledge"].get(symptom_name, 0) < self.importance_threshold
                        for disease_idx in current_top_diseases
                    )
                    if is_unimportant:
                        mask[i, symptom_idx] = 0
                
        mask_bool = mask == 1
        return mask_bool
        
