import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from numpy import linalg as LA
from components.episode_buffer import EpisodeBatch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from GRIN.grin import GRIN
from modules.auxiliary_nets import Err, Diff
from modules.LIDS.predict_net import Predict_Network_WithZ, Predict_Network, Predict_Z_obs_tau
from torch.distributions import kl_divergence, Normal
################################## set device ##########################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self,mac, state_dim, action_dim, has_continuous_action_space, action_std_init,args):
        super(ActorCritic, self).__init__()

        self.args=args
        self.grin=GRIN(args)

        self.mac=mac
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, args.ppo_hidden),
                            nn.Tanh(),
                            nn.Linear(args.ppo_hidden, args.ppo_hidden),
                            nn.Tanh(),
                            nn.Linear(args.ppo_hidden, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, args.ppo_hidden),
                            nn.Tanh(),
                            nn.Linear(args.ppo_hidden, args.ppo_hidden),
                            nn.Tanh(),
                            nn.Linear(args.ppo_hidden, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, args.ppo_hidden),
                        nn.Tanh(),
                        nn.Linear(args.ppo_hidden, args.ppo_hidden),
                        nn.Tanh(),
                        nn.Linear(args.ppo_hidden, 1)
                    )
        

        self.critic2 = nn.Sequential(
                        nn.Linear(args.z_embedding_dim, args.ppo_hidden),
                        nn.Tanh(),
                        nn.Linear(args.ppo_hidden, args.ppo_hidden),
                        nn.Tanh(),
                        nn.Linear(args.ppo_hidden, 1)
                    )

        
        


    def strategize(self, input):
        
        p_graph = self.grin.build_graph(input).to(self.device)
        p_res = self.grin.forward(p_graph)    
        z=p_res["loc_pred"]

        z_prob=p_res["zA_rv"]+p_res["zG_rv"] 

        return z, z_prob
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, input):

        
        state_values = self.critic(input)
        embed_input=self.mac.fc2(input)
        
        state_values2 = self.critic2(embed_input)
        vals=state_values+state_values2
        return  vals


class PPO:
    def __init__(self, batch:EpisodeBatch,mac, args, n_agents, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.mac=mac

        self.E=Err(self.args.h_dim,self.args.z_dim,self.args.predict_net_dim)
        self.F=Diff(self.args.h_dim,self.args.z_dim,self.args.predict_net_dim)

        self.n_agents=n_agents
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        self.args= args

        self.policy = ActorCritic(self.mac,self.args.h_dim,self.args.z_dim, has_continuous_action_space, action_std_init,args).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                        {'params': self.policy.critic2.parameters(), 'lr': lr_critic},
                        {'params': self.E.parameters(), 'lr': args.lr_alpha},
                        {'params': self.F.parameters(), 'lr': args.lr_eta}
                    ])

        self.policy_old = ActorCritic(self.mac,self.args.h_dim,self.args.z_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()



        ########      p(z|tau-,tau+)       #########

        self.eval_predict_Z = Predict_Z_obs_tau(
            args.rnn_hidden_dim, args.predict_net_dim, args.n_agents)
        self.target_predict_Z = Predict_Z_obs_tau(
            args.rnn_hidden_dim, args.predict_net_dim, args.n_agents)

        
        #######     

        self.eval_predict_withoutZ = Predict_Network(
                args.rnn_hidden_dim + args.obs_shape + args.n_actions, args.predict_net_dim, args.obs_shape)
        self.target_predict_withoutZ = Predict_Network(
            args.rnn_hidden_dim + args.obs_shape + args.n_actions, args.predict_net_dim, args.obs_shape)

        self.eval_predict_withZ = Predict_Network_WithZ(args.rnn_hidden_dim + args.obs_shape + args.n_actions + args.n_agents, args.predict_net_dim,
                                                            args.obs_shape, args.n_agents)
        self.target_predict_withZ = Predict_Network_WithZ(args.rnn_hidden_dim + args.obs_shape + args.n_actions + args.n_agents, args.predict_net_dim,
                                                            args.obs_shape, args.n_agents)

        

        if self.args.use_cuda:

            self.eval_predict_withZ.to(torch.device(self.args.GPU))
            self.target_predict_withZ.to(torch.device(self.args.GPU))

            self.eval_predict_withoutZ.to(torch.device(self.args.GPU))
            self.target_predict_withoutZ.to(torch.device(self.args.GPU))

            self.eval_predict_id.to(torch.device(self.args.GPU))
            self.target_predict_id.to(torch.device(self.args.GPU))

        self.target_predict_withZ.load_state_dict(
            self.eval_predict_withZ.state_dict())
        self.target_predict_withoutZ.load_state_dict(
            self.eval_predict_withoutZ.state_dict())
        self.target_predict_id.load_state_dict(
            self.eval_predict_id.state_dict())
            

    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def train_prob_nets(self, batch: EpisodeBatch, mac):
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = torch.cat([torch.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions
        



        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        initial_hidden = mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = torch.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        _, hidden_store, _ = mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach())
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        obs = batch["obs"][:, :-1]
        obs_next = batch["obs"][:, 1:]

        h_cat = hidden_store[:, :-1]
        add_id = torch.eye(self.args.n_agents).to(obs.device).expand(
            [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])

        p_graph = self.policy.grin.build_graph(h_cat).to(self.device)
        p_res = self.policy.grin.forward(p_graph)    
        z=p_res["loc_pred"]

        mask_reshape = mask.unsqueeze(-1).expand_as(
            h_cat[..., 0].unsqueeze(-1))

        _obs = obs.reshape(-1, obs.shape[-1]).detach()
        _obs_next = obs_next.reshape(-1, obs_next.shape[-1]).detach()
        _h_cat = h_cat.reshape(-1, h_cat.shape[-1]).detach()
        _add_id = add_id.reshape(-1, add_id.shape[-1]).detach()
        _z = z.reshape(-1, add_id.shape[-1]).detach()
        _mask_reshape = mask_reshape.reshape(-1, 1).detach()
        _actions_onehot = actions_onehot.reshape(
            -1, actions_onehot.shape[-1]).detach()


        h_cat_r = torch.cat(
                [torch.zeros_like(h_cat[:, 0]).unsqueeze(1), h_cat[:, :-1]], dim=1)
        intrinsic_input = torch.cat(
            [h_cat_r, obs, actions_onehot], dim=-1)
        _inputs = intrinsic_input.detach(
        ).reshape(-1, intrinsic_input.shape[-1])

        loss_withZ_list, loss_withoutZ_list, loss_predict_Z_list = [], [], []
        # update predict network
        for _ in range(self.args.predict_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(_obs.shape[0])), 256, False):
                loss_withoutZ = self.eval_predict_withoutZ.update(
                    _inputs[index], _obs_next[index], _mask_reshape[index])
                loss_withZ = self.eval_predict_withZ.update(
                    _inputs[index], _obs_next[index], _z[index], _mask_reshape[index])

                if loss_withoutZ:
                    loss_withoutZ_list.append(loss_withoutZ)
                if loss_withZ:
                    loss_withZ_list.append(loss_withZ)

        Z_for_predict = torch.tensor(self.list[0]).type_as(
            hidden_store).unsqueeze(0).unsqueeze(0)

        Z_for_predict = Z_for_predict.expand_as(hidden_store[..., 0])
        _Z_for_predict = Z_for_predict.reshape(-1)

        for _ in range(self.args.predict_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(_obs.shape[0])), 256, False):
                loss_predict_Z = self.eval_predict_Z.update(
                    _h_cat[index], _Z_for_predict[index], _mask_reshape[index].squeeze())
                if loss_predict_Z:
                    loss_predict_Z_list.append(loss_predict_Z)





    def _build_inputs(self, batch, t0 , t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        if self.args.obs_last_action:  # True for QMix
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][t0, t]))  # last actions are empty
            else:
                inputs.append(batch["actions_onehot"][t0, t - 1])
        inputs.append(batch["obs"][t0, t])  # b1av
        if self.args.obs_agent_id:  # True for QMix
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))  # onehot agent ID

        # inputs[i]: (bs,n,n)
        inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)  # (bs*n, act+obs+id)
        # inputs[i]: (bs*n,n); ==> (bs*n,3n) i.e. (bs*n,(obs+act+id))
        return inputs
    
    def update(self, ep_batch, t):
        

        agent_past_inputs = self._build_inputs(ep_batch, 0, t)  # (bs*n,(obs+act+id)) #########
        agent_future_inputs = self._build_inputs(ep_batch, t+1, t+self.args.segment_ratio*ep_batch.batch_size)
        avail_actions = ep_batch["avail_actions"][:, t]

        

        z, prob=self.policy.strategize(agent_past_inputs)


        e_alpha = self.E.forward(agent_past_inputs, z)
        Je = LA.norm(e_alpha) + self.args.lamda*LA.norm(agent_future_inputs - agent_past_inputs - self.F.forward(agent_past_inputs, z) - e_alpha)   #### Error loss
        




        ################# Rewards  ###############
        rewards = ep_batch["reward"][:, :-1]
        actions = ep_batch["actions"][:, :-1]
        terminated = ep_batch["terminated"][:, :-1].float()
        mask = ep_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = ep_batch["avail_actions"]
        actions_onehot = ep_batch["actions_onehot"][:, :-1]
        last_actions_onehot = torch.cat([torch.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions

        # Calculate estimated Q-Values  ####### For the Boltzman Operator
        self.mac.init_hidden(ep_batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = torch.cat((ep_batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        mac_out, hidden_store, local_qs = self.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach(),z)
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()



        # Calculate the Values necessary for the second Reward
        input_here_past = torch.cat((ep_batch["obs"][:t], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)
        input_here_future = torch.cat((ep_batch["obs"][t+1:t+self.args.segment_ratio*ep_batch.batch_size], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)
        self.target_mac.init_hidden(ep_batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]).to(self.args.device)
        z,_=self.meta_pol.strategize(initial_hidden_target)
        target_mac_out, _, _ = self.target_mac.agent.forward(
            input_here_past.clone().detach(), initial_hidden_target.clone().detach(),z)
        target_mac_out = target_mac_out[:, 1:]

        target_max_qvals_past = target_mac_out.max(dim=3)[0]

        target_mac_out, _, _ = self.target_mac.agent.forward(
            self.F.forward(agent_past_inputs, z), initial_hidden_target.clone().detach())
        target_mac_out = target_mac_out[:, 1:]

        target_max_qvals_future = target_mac_out.max(dim=3)[0]

        with torch.no_grad():
            R_m = target_max_qvals_past - target_max_qvals_future


            obs = ep_batch["obs"][:, :-1]
            obs_next = ep_batch["obs"][:, 1:]
            h_cat = hidden_store[:, :-1]
            add_id = torch.eye(self.args.n_agents).to(obs.device).expand(
                [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])

            if self.args.ifaddobs:
                h_cat_reshape = torch.cat(
                    [torch.zeros_like(h_cat[:, 0]).unsqueeze(1), h_cat[:, :-1]], dim=1)
                intrinsic_input = torch.cat(
                    [h_cat_reshape, obs, actions_onehot], dim=-1)
            else:
                intrinsic_input = torch.cat([h_cat, actions_onehot], dim=-1)

            z, prob=self.policy.strategize(intrinsic_input)

            log_p_o = self.target_predict_withoutZ.get_log_pi(
                intrinsic_input, obs_next)

            add_id = torch.eye(self.args.n_agents).to(obs.device).expand(
                [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])
            log_q_o = self.target_predict_withZ.get_log_pi(
                intrinsic_input, obs_next, z)
            obs_diverge = self.args.beta1 * log_q_o - log_p_o

            
            mac_out_c_list = []
            for item_i in range(self.args.n_agents):
                mac_out_c, _, _ = self.mac.agent.forward(
                    input_here[:, self.list[item_i]], initial_hidden,z)
                mac_out_c_list.append(mac_out_c)

            mac_out_c_list = torch.stack(mac_out_c_list, dim=-2)
            mac_out_c_list = mac_out_c_list[:, :-1]

            if self.args.ifaver:
                mean_p = torch.softmax(mac_out_c_list, dim=-1).mean(dim=-2)
            else:
                weight = self.target_predict_Z(h_cat)
                weight_expend = weight.unsqueeze(-1).expand_as(mac_out_c_list)
                mean_p = (weight_expend *
                          torch.softmax(mac_out_c_list, dim=-1)).sum(dim=-2)

            q_pi = torch.softmax(self.args.beta1 * mac_out[:, :-1], dim=-1)

            pi_diverge = torch.cat([(q_pi[:, :, z] * torch.log(q_pi[:, :, z] / mean_p[:, :, z])).sum(
                dim=-1, keepdim=True)], dim=-1).unsqueeze(-1)       ######### log [sigma / p(.| tau)]




            information_rewards = obs_diverge + self.args.beta2 * pi_diverge

            alltau=torch.cat([input_here_past,input_here_future], dim=-1)
            z_prob= self.eval_predict_Z.forward(alltau)

            information_rewards+=z_prob

            information_rewards = information_rewards.mean(dim=2)


            self.buffer.rewards.append(R_m + information_rewards)        ############

        # Monte Carlo estimate of returns#
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        agent_past_inputs = self._build_inputs(ep_batch, 0, t)  # (bs*n,(obs+act+id)) #########
        agent_future_inputs = self._build_inputs(ep_batch, t+1, t+self.args.segment_ratio*ep_batch.batch_size)
        avail_actions = ep_batch["avail_actions"][:, t]

        

        old_strats, old_probs=self.policy.strategize(agent_past_inputs)

        old_logprobs=torch.log(old_probs)

        agent_inputs = torch.squeeze(torch.stack(agent_past_inputs, dim=0)).detach().to(device)
        old_strats = torch.squeeze(torch.stack(old_strats, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(old_logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs,  dist_entropy = self.policy.strategize(agent_inputs, old_strats)

            state_values=self.policy.evaluate(agent_inputs)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy - self.args.lamda_e * Je
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


