import torch
from torch.distributions import kl_divergence

from agents.SAC import SoftActorCriticProperty, SACAlgo, SACAgentCore
from model.BaseManager import MLFeatures
from utils.standardization import rescale
from utils.util import to_tensors
from utils.util import get_module_device


class ROMASoftActorCriticProperty(SoftActorCriticProperty, MLFeatures):
    __RequiredTarget__ = {'Actor': False, 'DualCritic': True, 'Encoder': False, 'DisparityNet': False}

    @property
    def dis_net(self):
        return self._model_callbacks['DisparityNet']

    @property
    def critic_param_groups(self):
        sac_groups = super(ROMASoftActorCriticProperty, self).critic_param_groups
        return sac_groups + [self._param_groups['DisparityNet']]


class ROMASACAlgo(ROMASoftActorCriticProperty, SACAlgo):
    def __init__(self, model_callbacks, batch_size, alpha, buffer_capacity, optim_cls, optim_kwargs,
                 **other_kwargs):
        super(ROMASACAlgo, self).__init__(model_callbacks, batch_size, alpha, buffer_capacity, optim_cls, optim_kwargs,
                                          **other_kwargs)
        self.dis_time = 0
        self.h_loss_weight = 0.01
        self.kl_loss_weight = 1e-4
        self.dis_loss_weight = 1e-4
        self.soft_constraint_weight = 1.
        self.max_ce_loss = 2e3

        self.mutual_info_bias = +13.9
        self.dis_loss_weight_schedule = lambda t: self.dis_loss_weight if t > self.dis_time else 0.

        # def dis_loss_weight_schedule_sigmoid(self, t_glob):
        #     return self.args.dis_loss_weight / (1 + math.exp((1e7 - t_glob) / 2e6))

    def _process_buffer(self, full_state, action, reward, done, gamma=None, init_states=None, **info):
        def split_and_merge(hs, role, dim=-2, n_step=None):
            if n_step is None:
                n_step = 1
            idx = torch.arange(hs.size(dim))
            idx = torch.cat((idx[:-n_step], idx[n_step:])).to(hs.device)
            h_cur, h_next = hs.index_select(dim, idx).chunk(2, -2)
            cur_role, next_role = role.index_select(dim, idx).chunk(2, -2)
            return torch.cat((h_cur, cur_role), -1), torch.cat((h_next, next_role), -1)

        if init_states is not None:
            # (*, Time, State) -> (GRULayer*Direction, Batch, HiddenState)
            with torch.no_grad():
                _, init_states, _ = self._encode_state(init_states, deterministic=True)

        full_h_state, _, (role_embed, _, dist_embed, dist_infer) = \
            self._encode_state(full_state, h_in=init_states, train_mode=True)
        h_state, h_next_state = split_and_merge(full_h_state, role_embed)
        gamma = torch.stack(gamma.repeat(*done.shape[1:], 1).unbind(-1))
        info.update(gamma=gamma, role_embed=role_embed, dist_embed=dist_embed, dist_infer=dist_infer)
        return h_state, action, reward, h_next_state, done, info

    def compute_loss_q(self, state, action, reward, next_state, done, **info):
        # Note: role_embed is included in state
        loss_q, q_info = super(ROMASACAlgo, self).compute_loss_q(state, action, reward, next_state, done, **info)
        loss_latent = self.compute_loss_latent(**info)
        if self.time_step % self.log_interval == 0:
            print("time step:{} q loss:{} = loss_q:{} + loss_latent:{}"
                  .format(self.time_step, loss_q + loss_latent, loss_q, loss_latent))
        return loss_q + loss_latent, q_info

    def compute_loss_latent(self, role_embed, dist_embed, dist_infer, valid_mask=None, **other_kwargs):
        """

        :param role_embed: torch.FloatTensor (*, Time, Role)
        :param dist_embed: torch.distribution.Normal
        :param dist_infer: torch.distribution.Normal
        :param valid_mask: (*, Time)
        :param other_kwargs:
        :return:
        """
        loss_latent = self.cross_entropy(dist_embed, dist_infer)
        cur_dis_loss_weight = self.dis_loss_weight_schedule(self.time_step)
        if cur_dis_loss_weight > 0.:
            cd_loss = self.cross_disparity(role_embed, dist_embed, agent_dim=-3)
            loss_latent = loss_latent + cd_loss * cur_dis_loss_weight
        if valid_mask:
            # Use Mask to take mean along time dim
            time_length = valid_mask.sum(-1, keepdims=True)
            loss_latent = loss_latent.sum(-2) / time_length
        loss_latent = loss_latent.mean()
        return loss_latent

    def cross_entropy(self, dist_embed, dist_infer):
        # Cross Entropy Loss = Entropy + KL-Divergence
        # (*, Latent) -> (*, 1)
        entropy = self.h_loss_weight * dist_embed.entropy().sum(-1, keepdim=True)  # 'BatchMean'
        kl_div = self.kl_loss_weight * kl_divergence(dist_embed, dist_infer).sum(-1, keepdim=True)  # 'BatchMean'
        loss = entropy + kl_div
        loss_factor = self.max_ce_loss / loss.mean()
        if loss_factor < 1.:
            loss *= loss_factor
        ce_loss = torch.log(1 + torch.exp(loss))
        return ce_loss

    def cross_disparity(self, role_embed, dist_embed, agent_dim, **kwargs):
        # TODO mask
        """

        :param role_embed: shape = (*, Agent, *, Latent)
        :param dist_embed: mean shape = (*, Agent, *, Latent)
        :param agent_dim: Agent Dimension Location
        :return:
        """
        # Copy & Reshape Latent

        agent_dim = agent_dim + role_embed.dim() if agent_dim < 0 else agent_dim
        assert 0 <= agent_dim < role_embed.dim()
        n_agent = role_embed.size(agent_dim)
        role_embed = role_embed.unsqueeze(agent_dim + 1)

        src_latent = role_embed.repeat_interleave(n_agent, agent_dim + 1)
        tgt_latent = src_latent.transpose(agent_dim, agent_dim + 1)

        lead_dims = src_latent.shape[:-1]
        eps = 1e-12

        def _rescale(x, dim):
            # If take max+eps - min, smaller digit will be omitted
            # So we take max-min first, then plus eps
            return rescale(x, x.min(dim, keepdim=True)[0], x.max(dim, keepdim=True)[0], eps=eps)

        # Mutual Info
        # (*, Agent, Other, Time, Latent) -> (*, Agent * Other, Time, 1)
        mi_in = torch.stack(src_latent.unbind(agent_dim), dim=0)
        mi = torch.clamp(dist_embed.log_prob(mi_in) + self.mutual_info_bias, min=-self.mutual_info_bias)  # TODO wtf?
        mi = torch.stack(mi.unbind(dim=0), dim=agent_dim).flatten(agent_dim, agent_dim + 1).mean(-1, keepdim=True)
        mi = _rescale(mi, agent_dim)

        # Role Disparity
        # (*, Agent, Other, Time, Latent) -> (*, Agent * Other, Time, 1)
        latent_pairs = torch.cat((src_latent, tgt_latent), dim=-1)  # cat(src_agent, tgt_agent)
        disparity = self.dis_net(latent_pairs)
        disparity = disparity.abs().flatten(agent_dim, agent_dim + 1)
        disparity = _rescale(disparity, agent_dim)

        # Remove Other dim.
        # (*, Agent * Other, Time, 1) -> (*, Agent, Time, 1)
        dis_loss = torch.clamp(mi + disparity, max=1.0).reshape(*lead_dims, 1).mean(dim=agent_dim + 1)
        dis_norm = torch.norm(disparity, p=1, dim=agent_dim, keepdim=True)
        dis_norm = dis_norm.repeat_interleave(n_agent, agent_dim) / disparity.size(agent_dim)  # Mean along Agent Adj.

        cd_loss = dis_norm - self.soft_constraint_weight * dis_loss

        return cd_loss


class ROMASACAgentCore(ROMASoftActorCriticProperty, SACAgentCore):
    __AlgorithmArch___ = ROMASACAlgo

    def __init__(self, observation_spaces, action_dim, reward_fn=None, **kwargs):
        """

        :param kwargs: Detail see SACAgentCore
        """
        super(ROMASACAgentCore, self).__init__(observation_spaces, action_dim, reward_fn=reward_fn, **kwargs)
        if reward_fn is None:
            self.reward_func = lambda reward, state, *args, **other: (reward, [reward.max(-1), reward.min(-1)])
        self.trajectory_state = None  # Update at _encode_state

    def _encode_state(self, state, h_in=None, **kwargs):
        if h_in is None:
            h_in = self.trajectory_state
        state = to_tensors(self.state_fn(state), device=get_module_device(self._model_callbacks))
        hidden_state, self.trajectory_state, role_embed = self.encoder(state.unsqueeze(0), h_in=h_in, **kwargs)
        hidden_state = torch.cat((hidden_state, role_embed), dim=-1).squeeze(0)
        return hidden_state

    def reset(self):
        self.trajectory_state = None
        super(ROMASACAgentCore, self).reset()
