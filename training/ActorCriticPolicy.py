from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium
from typing import Callable
from torch import nn
import torch
from typing import Tuple


class CustomNetwork(nn.Module):
    def __init__(
        self, feature_dim: int, last_layer_dim_pi: int, last_layer_dim_vf: int
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 192), nn.ReLU(), nn.Linear(192, 192), nn.ReLU()
        )
        self.policy_net = nn.Linear(192, last_layer_dim_pi)
        self.value_net = nn.Linear(192, last_layer_dim_vf)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.action_space.n, 1)
