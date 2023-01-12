from screenrecorder import *
from PPOLib import *

import win32process
#stolen from website http://timgolden.me.uk/python/win32_how_do_i/find-the-window-for-my-subprocess.html
def get_hwnds_for_pids(pids):
  def callback (hwnd, hwnds):
    if win32gui.IsWindowVisible (hwnd) and win32gui.IsWindowEnabled (hwnd):
      _, found_pid = win32process.GetWindowThreadProcessId (hwnd)
      if found_pid in pids:
        hwnds[found_pid] = hwnd
    return True
    
  hwnds = {}
  win32gui.EnumWindows (callback, hwnds)
  return hwnds
class DGames:
    def __init__(self,interval,device) -> None:
        self.device = device
        self.pids = []
        self.observation_space = (1,214,120)
        for proc in psutil.process_iter():
            if "Downwell" in proc.name():
                self.pids.append(proc.pid)
        if len(self.pids)==0:
            raise Exception("failed to find dowenwll")
        self.hwnds = get_hwnds_for_pids(self.pids)
        self.games = [DGame(interval,None,True,device,pid,self.hwnds[pid]) for pid in self.pids]
        self.env_count = len(self.games)
        self.pause_games()
    def move(self,actions):
        f_obs = torch.zeros((self.env_count,)+self.observation_space)
        f_rewards = torch.zeros(self.env_count)
        f_dones = torch.zeros_like(f_rewards)
        threads = []
        for i,g in enumerate(self.games):
            state,reward,is_loss,t= g.move(int(actions[i]))
            f_obs[i]=state.feed
            f_rewards[i] = reward
            f_dones[i] = is_loss
            threads.append(t)
        return f_obs.to(self.device),f_rewards.to(self.device),f_dones.to(self.device),threads
    def make_observation(self):
        f_obs = torch.zeros((self.env_count,)+self.observation_space)
        for i,g in enumerate(self.games):
            f_obs[i] = g.get_state().feed

        return f_obs.to(self.device)
    def pause_games(self):
        for g in self.games:
            g.pause_game()
    def unpause_games(self):
        for g in self.games:
            g.unpause_game()
    def quit(self):
        for g in self.games:
            g.process.kill()
    def handle_death(self,dones,threads):
        #I cant do this yet ^^^^ 
        new_threads = []
        a=[]
        for i,d in enumerate(dones):
            if d ==1:
                t=threading.Thread(target=DGame.restart_loss,args=(self.games[i],))
                new_threads.append(t)
                a.append(i)
        for i,d in enumerate(dones):
            if d==1:
                threads[i].join()
                self.games[i].unpause_game()
        for t in new_threads:
            t.start()
            t.join()
        [self.games[i].pause_game() for i in a]
    def mass_restart(self):
        a=[]
        for g in self.games:
            t = threading.Thread(target=DGame.restart,args=(g,))
            t.start()
            a.append(t)
        [t.join() for t in a]


class DownwellAgent(PrevAgent):
    def __init__(self, action_shape, device):
        super(PrevAgent,self).__init__()
        self.device = device
        self.action_shape = action_shape
        self.network= nn.Sequential(
            nn.Conv2d(1,32,8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(16192,512)),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(512+action_shape,128)
        for n,p in self.lstm.named_parameters():
            if "bias" in n:
                nn.init.constant_(p,0)
            elif "weight" in n:
                nn.init.orthogonal_(p,1.0)
        self.actor = init_layer(nn.Linear(128,action_shape),std=.01)
        self.critic = init_layer(nn.Linear(128,1),std=1) 
def play():
    device = torch.device("cuda")

    game = DGames(.1,device)
   # game.quit()
    game.unpause_games()
    
    game.mass_restart()
    time.sleep(2.1)
    game.pause_games()
    mem_size = 128
    action_space = 6
    model =DownwellAgent(action_space,device).to(device)
    model_name = "considers_last_action"
    try:
        model.load_state_dict(torch.load(model_name))
    except:
        print("Error loading model")

    adam = torch.optim.Adam(model.parameters(),lr=2.5e-4,eps=1e-5) 
    prev_lstm_state =  (torch.zeros((model.lstm.num_layers,game.env_count,model.lstm.hidden_size)).to(device),torch.zeros((model.lstm.num_layers,game.env_count,model.lstm.

                    hidden_size)).to(device))
    memories = Memories(game.env_count,mem_size,game.observation_space,action_space,prev_lstm_state,device)
    prev_observation = game.make_observation()
    prev_done = torch.zeros(game.env_count,device=device)
    threads = []
    prev_actions = torch.full((game.env_count,),5,device=device)
    while True:
        memories.init_lstm_states=  (prev_lstm_state[0].clone(),prev_lstm_state[1].clone())
        starting_actions = prev_actions.clone()
        for i in range(mem_size):
            memories.observations[i] = prev_observation
            memories.is_dones[i] = prev_done
            with torch.no_grad():
                actions,log_probs, _ ,value,prev_lstm_state = model.forward(prev_observation,prev_lstm_state,prev_done,prev_actions) 
                memories.values[i] = value.flatten()
                memories.log_probs[i] =log_probs
                memories.actions[i] = actions
            # print(value)
            prev_actions = actions
            prev_observation,rewards,prev_done,threads = game.move(actions)
            # print("reward:",rewards)

            game.handle_death(prev_done,threads)
            [t.join() for t in threads]


            memories.rewards[i] = rewards
        print("total rewards:",memories.rewards.sum())
        print("total deaths:",memories.is_dones.sum())
        train_ppo(4,adam,model,memories,prev_observation,prev_done,prev_lstm_state,starting_actions)
        torch.save(model.state_dict(),model_name)
if __name__=="__main__":
    play()
