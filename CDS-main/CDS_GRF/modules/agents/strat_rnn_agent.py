import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent_with_strategy(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent_with_strategy, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.z_fc1 = nn.Linear(args.z_dim + args.n_agents, args.z_embedding_dim)
        self.z_fc2 = nn.Linear(args.z_embedding_dim, args.z_embedding_dim)
        self.z_fc3 = nn.Linear(args.z_embedding_dim, args.n_actions)

        self.hyper = True
        self.hyper_z_fc1 = nn.Linear(args.z_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)

        self.fc2_common = nn.Linear(args.rnn_hidden_dim +args.z_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, z):
        agent_ids = th.eye(self.args.n_agents, device=inputs.device).repeat(z.shape[0], 1)
        z_repeated = z.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        z_input = th.cat([z_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_z_fc1(z_input).reshape(-1, self.args.n_actions, self.args.rnn_hidden_dim)
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.z_fc1(z_input))
            z = F.tanh(self.z_fc2(z))
            wz = self.z_fc3(z)

            wq = q * wz

        q_c_input=th.cat([h,z], dim=-1)
        q_common= self.fc2_common(q_c_input)


        wq+=q_common


        return wq, h