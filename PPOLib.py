import torch
import numpy as np
from torch import nn
from torch.distributions.categorical import Categorical
class Memories:
    def __init__(self,env_size,mem_size,obs_shape,action_size,init_lstm_states,device) -> None:
        self.mem_size = mem_size
        self.device = device
        self.env_size = env_size
        self.init_lstm_states = init_lstm_states
        self.observations = torch.zeros((mem_size,env_size,)+obs_shape).to(device)
        self.rewards = torch.zeros((mem_size,env_size)).to(device)
        self.is_dones = torch.zeros((mem_size,env_size)).to(device)
        self.values = torch.zeros((mem_size,env_size)).to(device)
        self.log_probs= torch.zeros((mem_size,env_size)).to(device)
        self.actions = torch.zeros((mem_size,env_size,)).to(device).to(torch.int64)
        self.first_actions = torch.zeros(env_size).to(device)

def init_layer(layer,std=np.sqrt(2),bias_const=0):
    torch.nn.init.orthogonal_(layer.weight,std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias,bias_const)
    return layer
layer_init = init_layer
class Agent(nn.Module):
    def __init__(self,action_shape,device):
        self.device = device
        super().__init__()

        self.network= nn.Sequential(
            nn.Conv2d(1,32,8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(46592,512)),
            nn.ReLU()

        )
    # itself 
        self.lstm = nn.LSTM(512,128)
        for n,p in self.lstm.named_parameters():
            if "bias" in n:
                nn.init.constant_(p,0)
            elif "weight" in n:
                nn.init.orthogonal_(p,1.0)
        self.actor = init_layer(nn.Linear(128,action_shape),std=.01)
        self.critic = init_layer(nn.Linear(128,1),std=1) 
    def get_hidden(self,x,lstm_state,done):
        hidden = self.network(x )

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


        hiddens = self.network(x)
        batch_size = lstm_state[0].shape[1]
        sequence_length = x.shape[0]//batch_size
        results = torch.zeros((sequence_length,batch_size,self.lstm.hidden_size)).to(self.device)
        i=0
        hiddens = hiddens.reshape((-1,batch_size,self.lstm.input_size))
        is_dones = is_dones.reshape((-1,batch_size))
        for h,done in zip(hiddens,is_dones):
            r,lstm_state= self.lstm(h.unsqueeze(0),((1.0-done.view(1,-1,1))*lstm_state[0],(1.0-done.view(1,-1,1))*lstm_state[1]))
            results[i] = r[0]
            i+=1
        
        return results.reshape(x.shape[0],self.lstm.hidden_size),lstm_state

    def get_value(self,x,lstm_state,is_dones):
        hidden,_ = self.get_hidden(x,lstm_state,is_dones)
        return self.critic(hidden)

    def forward(self,x,lstm_state,is_dones,actions=None):
        #value, log_probs
        hidden,lstm_state = self.get_hidden(x,lstm_state,is_dones)
        actor= self.actor(hidden)

        probs = Categorical(logits=actor)
        
        if actions is None:
            actions= probs.sample()
        
        critic = self.critic(hidden)
        # print("value:",critic)
        # print("policy:",actor)
        return actions,probs.log_prob(actions),probs.entropy(),critic,lstm_state
class PrevAgent(nn.Module):
    def __init__(self,action_shape,device):
        self.device = device
        self.action_shape = action_shape
        super().__init__()

        self.network= nn.Sequential(
            nn.Conv2d(1,32,8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(46592,512+action_shape)),
            nn.ReLU()

        )
    # itself 
        self.lstm = nn.LSTM(512+action_shape,128)
        for n,p in self.lstm.named_parameters():
            if "bias" in n:
                nn.init.constant_(p,0)
            elif "weight" in n:
                nn.init.orthogonal_(p,1.0)
        self.actor = init_layer(nn.Linear(128,action_shape),std=.01)
        self.critic = init_layer(nn.Linear(128,1),std=1) 
    def get_hidden(self,x,lstm_state,done,prev_actions):
        hidden = self.network(x )

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        prev_actions = torch.nn.functional.one_hot(prev_actions,num_classes=self.action_shape)
        hidden = torch.cat((hidden,prev_actions),dim=1)
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


        hiddens = self.network(x)
        batch_size = lstm_state[0].shape[1]
        sequence_length = x.shape[0]//batch_size
        results = torch.zeros((sequence_length,batch_size,self.lstm.hidden_size)).to(self.device)
        i=0
        hiddens = hiddens.reshape((-1,batch_size,self.lstm.input_size))
        is_dones = is_dones.reshape((-1,batch_size))
        for h,done in zip(hiddens,is_dones):
            r,lstm_state= self.lstm(h.unsqueeze(0),((1.0-done.view(1,-1,1))*lstm_state[0],(1.0-done.view(1,-1,1))*lstm_state[1]))
            results[i] = r[0]
            i+=1
        
        return results.reshape(x.shape[0],self.lstm.hidden_size),lstm_state

    def get_value(self,x,lstm_state,is_dones,prev_actions):
        hidden,_ = self.get_hidden(x,lstm_state,is_dones,prev_actions)
        return self.critic(hidden)

    def forward(self,x,lstm_state,is_dones,prev_actions,actions=None):
        #value, log_probs
        hidden,lstm_state = self.get_hidden(x,lstm_state,is_dones,prev_actions)
        actor= self.actor(hidden)

        probs = Categorical(logits=actor)
        
        if actions is None:
            actions= probs.sample()
        
        critic = self.critic(hidden)
        # print("value:",critic)
        # print("policy:",actor)
        return actions,probs.log_prob(actions),probs.entropy(),critic,lstm_state

def train_ppo(epochs,optimizer,model,memories:Memories,final_observations,final_dones,final_lstm_states,starting_actions):
 
    epsilon = .1
    gae_lambda = .95
    gamma = .99
    
    with torch.no_grad():
        advantages = torch.zeros_like(memories.rewards,device=memories.device)
        last_advantage = 0
        for i in reversed(range(memories.mem_size)):
            if memories.mem_size-1==i:
                is_next_done_multiplier =1-final_dones
                next_value = model.get_value(final_observations,final_lstm_states,final_dones,memories.actions[-1]).flatten()
            else:
                is_next_done_multiplier = 1-memories.is_dones[i+1]
                next_value = memories.values[i+1]
                
            delta = memories.rewards[i]+gamma*next_value*is_next_done_multiplier-memories.values[i]
            advantages[i] = delta+gamma*gae_lambda*is_next_done_multiplier*last_advantage
            last_advantage= advantages[i]
        returns =advantages+memories.values
    envs_per_batch = 2
    indices = np.arange(memories.env_size*memories.mem_size).reshape(memories.mem_size,memories.env_size)
    
    prev_actions = torch.cat((starting_actions.unsqueeze(0),memories.actions[:-1,:]))
    f_prev_actions = prev_actions.reshape(-1)
    f_obs = memories.observations.reshape((-1,)+memories.observations.shape[2:])
    f_log_probs = memories.log_probs.reshape(-1)
    f_actions = memories.actions.reshape(-1)
    f_dones = memories.is_dones.reshape(-1)
    f_advantages = advantages.reshape(-1)
    f_returns = returns.reshape(-1)
    f_values = memories.values.reshape(-1)
    flat_indices = np.arange(memories.env_size)

    for epoch in range(epochs):
        np.random.shuffle(flat_indices)
        for k,i in enumerate(range(0,memories.env_size,envs_per_batch)):
            env_indices= flat_indices[i:min(i+envs_per_batch,memories.env_size)]
            unraveled_indices = indices[:,env_indices].ravel()


            action, log_prob, entropy, new_value, final_lstm= model.forward(f_obs[unraveled_indices],
                (memories.init_lstm_states[0][:,env_indices],memories.init_lstm_states[1][:,env_indices]),
                f_dones[unraveled_indices],
                f_prev_actions[unraveled_indices],
                f_actions[unraveled_indices])

            ratio = (log_prob-f_log_probs[unraveled_indices]).exp()

            batch_advantages =f_advantages[unraveled_indices]
            batch_advantages=  (batch_advantages-batch_advantages.mean())/(batch_advantages.std()+1e-8)

            
            policy_loss_no_clip =- batch_advantages*ratio
            policy_loss_clip = -batch_advantages*torch.clamp(ratio,1-epsilon,1+epsilon)
            policy_loss = torch.max(policy_loss_clip,policy_loss_no_clip).mean()


            new_value = new_value.flatten()
            value_loss_no_clip = (new_value-f_returns[unraveled_indices])**2
            clipped =f_values[unraveled_indices]+torch.clamp(new_value-f_values[unraveled_indices],-epsilon,epsilon)
            value_loss_clip = (f_returns[unraveled_indices]-clipped)**2
            value_loss = .5*torch.max(value_loss_clip,value_loss_no_clip).mean()

            entropy_loss = entropy.mean()
            vf_coeff = .5
            ent_coeff = .01
            loss = policy_loss - ent_coeff * entropy_loss+ value_loss * vf_coeff
            # print("vloss:",value_loss)
            # print("ploss:",policy_loss)
            # print("loss:",loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step()
#