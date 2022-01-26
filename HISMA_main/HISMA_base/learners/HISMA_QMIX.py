import copy
import torch as th
import numpy as np
import torch.nn.functional as F
from numpy import linalg as LA
from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
from components.episode_buffer import EpisodeBatch
from meta_policy.PPO import PPO

class HISMA_QMIX:
    def __init__(self, mac, scheme, logger, args, meta_pol):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.meta_pol=meta_pol

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = QMixer(args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.list = [(np.arange(args.n_agents - i) + i).tolist() + np.arange(i).tolist()
                     for i in range(args.n_agents)]

    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions


        




        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        z,_=self.meta_pol.strategize(input_here)
        mac_out, hidden_store, local_qs = self.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach(),z)
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals -
                      chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]).to(self.args.device)
        z,_=self.meta_pol.strategize(initial_hidden_target)
        target_mac_out, _, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach(),z)
        target_mac_out = target_mac_out[:, 1:]

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Surprise IM
        with th.no_grad():


            obs = batch["obs"][:, :-1]
            obs_next = batch["obs"][:, 1:]
            h_cat = hidden_store[:, :-1]
            add_id = th.eye(self.args.n_agents).to(obs.device).expand(
                [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])

            if self.args.ifaddobs:
                h_cat_reshape = th.cat(
                    [th.zeros_like(h_cat[:, 0]).unsqueeze(1), h_cat[:, :-1]], dim=1)
                intrinsic_input = th.cat(
                    [h_cat_reshape, obs, actions_onehot], dim=-1)
            else:
                intrinsic_input = th.cat([h_cat, actions_onehot], dim=-1)

            z, prob=self.policy.strategize(intrinsic_input)

            log_p_o = self.target_predict_withoutZ.get_log_pi(
                intrinsic_input, obs_next)

            add_id = th.eye(self.args.n_agents).to(obs.device).expand(
                [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])
            log_q_o = self.target_predict_withZ.get_log_pi(
                intrinsic_input, obs_next, z)
            obs_diverge = self.args.beta1 * log_q_o - log_p_o

            
            mac_out_c_list = []
            for item_i in range(self.args.n_agents):
                mac_out_c, _, _ = self.mac.agent.forward(
                    input_here[:, self.list[item_i]], initial_hidden,z)
                mac_out_c_list.append(mac_out_c)

            mac_out_c_list = th.stack(mac_out_c_list, dim=-2)
            mac_out_c_list = mac_out_c_list[:, :-1]

            if self.args.ifaver:
                mean_p = th.softmax(mac_out_c_list, dim=-1).mean(dim=-2)
            else:
                weight = self.target_predict_Z(h_cat)
                weight_expend = weight.unsqueeze(-1).expand_as(mac_out_c_list)
                mean_p = (weight_expend *
                          th.softmax(mac_out_c_list, dim=-1)).sum(dim=-2)

            q_pi = th.softmax(self.args.beta1 * mac_out[:, :-1], dim=-1)

            pi_diverge = th.cat([(q_pi[:, :, z] * th.log(q_pi[:, :, z] / mean_p[:, :, z])).sum(
                dim=-1, keepdim=True)], dim=-1).unsqueeze(-1)       ######### log [sigma / p(.| tau)]


            input_here_past = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)
            input_here_future = th.cat((batch["obs"][:-1], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

            information_rewards = obs_diverge + self.args.beta2 * pi_diverge

            alltau=th.cat([input_here_past,input_here_future], dim=-1)
            z_prob= self.eval_predict_Z.forward(alltau)

            information_rewards+=z_prob

            information_rewards = information_rewards.mean(dim=2)
            Residual_error=LA.norm(input_here_future- self.F.forward(input_here_past, z)) 

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards -self.args.alpha_info*information_rewards +self.args.beta_residual*Residual_error + \
            self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            if self.mixer == None:
                tot_q_data = np.mean(tot_q_data, axis=2)
                tot_target = np.mean(tot_target, axis=2)

            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        update_prior = (masked_td_error ** 2).squeeze().sum(dim=-1,
                                                            keepdim=True) / mask.squeeze().sum(dim=-1, keepdim=True)

        

        norm_loss = F.l1_loss(local_qs, target=th.zeros_like(
            local_qs), reduction='none')[:, :-1]
        mask_expand = mask.unsqueeze(-1).expand_as(norm_loss)
        norm_loss = (norm_loss * mask_expand).sum() / mask_expand.sum()
        loss += 0.1 * norm_loss

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                 mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(
                "target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            
            self.log_stats_t = t_env

        return update_prior.squeeze().detach()

    
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(th.device(self.args.GPU))
            self.target_mixer.to(th.device(self.args.GPU))

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
