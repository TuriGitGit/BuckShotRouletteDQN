import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import multiprocessing as mp
from multiprocessing import Queue, Event

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using: {device}")

class Game():
    def __init__(self):
        self.resetGame()
    
    def resetShells(self):
        """Adds a random number of live and blank shells to the shotgun."""
        self.live_shells, self.blank_shells = random.randint(1, 4), random.randint(1, 4)
        self.shells = self.totalShells()
        self.current_round_num = self.shell = 0
        
    def restockItems(self):
        """Restocks the round for the AI and DEALER."""
        self.AI_items = list(filter(None, self.AI_items))
        self.DEALER_items = list(filter(None, self.DEALER_items))
        for _ in range(4):
            if len(self.AI_items) < 8: self.AI_items.append(random.randint(1, 6))
            if len(self.DEALER_items) < 8: self.DEALER_items.append(random.randint(1, 6))
            
        while len(self.AI_items) < 8: self.AI_items.append(0)
        while len(self.DEALER_items) < 8: self.DEALER_items.append(0)
    
    def totalShells(self): return self.live_shells + self.blank_shells
    
    def determineShell(self): return 1 if random.random() <= (self.live_shells / self.shells) else 0.5
    
    def riggedDetermine(self, live: bool): return 1 if live else 0.5
    
    def removeShell(self, live: bool):
        if live: self.live_shells -= 1
        else: self.blank_shells -= 1
    
    def outOfShells(self):
        self.resetShells()
        self.restockItems()
    
    def resetGame(self):
        """Resets the game state, initializes the shotgun, and loads bullets."""
        self.resetShells()
        self.AI_items = []
        self.DEALER_items = []  # 0:nothing, 1:beer 2:magnifier 3:smoke 4:inverter 5:cuffs 6:saw
        self.AI_hp = self.DEALER_hp = 4
        self.AI_can_play = self.DEALER_can_play = True
        self.invert_odds = self.is_sawed = False
        self.restockItems()
    
    def debugPrintGame(self):
        """Prints the current game state for debugging and visualization."""
        print(f"AI HP: {self.AI_hp}, DEALER HP: {self.DEALER_hp}")
        print(f"Live Shells: {self.live_shells}, Blank Shells: {self.blank_shells}")
        print(f"AI Items: {self.AI_items}")
        print(f"DEALER Items: {self.DEALER_items}")
        print(f"Current Round Number: {self.current_round_num}")
        print(f"Is sawed?: {self.is_sawed}")
        print(f"Invert Odds?: {self.invert_odds}")
        #WIP
    
    def removeUnknownShell(self):
        if random.randint(0, 1) == 1 and self.live_shells > 0: self.live_shells -= 1
        else: self.blank_shells -= 1
            
    def drinkBeer(self, player: bool = False):
        """Player drinks beer, returns reward."""
        if self.totalShells == 1: self.outOfShells(); return 0.2
        if player:
            if 1 in self.AI_items:
                self.AI_items.remove(1)
                self.AI_items.append(0)
                if self.shell == 0: self.removeUnknownShell()
                elif self.shell == 1: self.live_shells -= 1
                else: self.blank_shells -= 1
                return 0.2
            else: return -1
        elif 1 in self.DEALER_items: 
            self.DEALER_items.remove(1)
            self.DEALER_items.append(0)
            self.removeUnknownShell()
            
    def magnifier(self, player: bool = False):
        """Player breaks glass, determining self.shell, returns reward."""
        if player:
            if 2 in self.AI_items:
                self.AI_items.remove(2)
                self.AI_items.append(0)
                self.shell = self.determineShell()
                return 1
            else: return -1
        elif 2 in self.DEALER_items: 
            self.DEALER_items.remove(2)
            self.DEALER_items.append(0)
    
    def smoke(self, player: bool = False):
        """Player smokes, regains 1 hp, returns reward."""
        if player:
            
            if 3 in self.AI_items:
                self.AI_items.remove(3)
                self.AI_items.append(0)
                self.AI_hp = min(4, self.AI_hp+1)
                return 1
            else: return -1
        elif 3 in self.DEALER_items:
                self.DEALER_items.remove(3)
                self.DEALER_items.append(0)
                self.DEALER_hp = min(4, self.DEALER_hp+1)
    
    def invert(self):
            if self.shell == 0.5: self.shell = 1
            elif self.shell == 1: self.shell = 0.5
            else: self.invert_odds = True
            
    def inverter(self, player: bool = False):
        """Player inverts current round, returns reward."""
        if player:
            if 4 in self.AI_items:
                self.AI_items.remove(4)
                self.AI_items.append(0)
                if self.shell == 0.5: self.blank_shells -= 1; self.live_shells += 1
                elif self.shell == 1: self.blank_shells += 1; self.live_shells -= 1
                self.invert()
                return 0.2
            else: return -1
        elif 4 in self.DEALER_items: 
            self.DEALER_items.remove(4)
            self.DEALER_items.append(0)
    
    def cuff(self, player: bool = False):
        """Player cuffs opponent, skipping their turn, returns reward."""
        if player:
            if 5 in self.AI_items:
                self.AI_items.remove(5)
                self.AI_items.append(0)
                self.DEALER_can_play = False
                return 1
            else: return -1
        elif 5 in self.DEALER_items:
                self.DEALER_items.remove(5)
                self.DEALER_items.append(0)
                self.AI_can_play = False
    
    def saw(self, player: bool = False):
        """Player saws off shotgun, doubling damage, returns reward."""
        if player:
            if 6 in self.AI_items:
                self.AI_items.remove(6)
                self.AI_items.append(0)
                self.is_sawed = True
                return 1 if self.shell != 0.5 else -2
            else: return -1
        elif 6 in self.DEALER_items:
                self.DEALER_items.remove(6)
                self.DEALER_items.append(0)
                self.is_sawed = True
    
    def AIshootAI(self):
        """Determines the outcome of the shot if not already known, and shoots AI, returns reward."""
        if self.shell == 0:
            self.shell = self.determineShell()
            if self.shell == 1:
                self.AI_hp -= 1 if not self.is_sawed else 2
                return -3 if not self.is_sawed else -6 
            else: return 0 if not self.is_sawed else -2
        elif self.shell == 0.5: 
            return 2 if not self.is_sawed else -8
        elif not self.is_sawed: self.AI_hp -= 1; return -20
        else: self.AI_hp -= 2; return -40
        
    def AIshootDEALER(self):
        """Determines the outcome of the shot if not already known, and shoots DEALER, returns reward."""
        if self.shell == 0:
            self.shell = self.determineShell()
            if self.shell == 1:
                self.live_shells -= 1
                if not self.is_sawed: self.DEALER_hp -= 1; return 3
                else: self.DEALER_hp -= 2; return 6
            else: 
                self.blank_shells -= 1
                return 0
        elif self.shell == 1:
            self.live_shells -= 1
            if not self.is_sawed: self.DEALER_hp -= 1; return 4
            else: self.DEALER_hp -= 2; return 8
        else: return -20 if not self.is_sawed else -32
    
    def DEALERshootDEALER(self):
        """Determines the outcome of the shot if not already known, and shoots DEALER."""
        if self.shell == 0.5: self.AI_can_play = False
        elif self.shell == 0: self.shell = self.determineShell()
        elif self.shell == 1: self.DEALER_hp -= 1
    
    def DEALERshootAI(self):
        """Determines the outcome of the shot if not already known, and shoots AI."""
        if self.shell == 1: self.AI_hp -= 1 if not self.is_sawed else 2
        elif self.shell == 0 and self.determineShell() == 1: self.AI_hp -= 1 if not self.is_sawed else 2
        
    def DEALERSmoke(self):
        """The DEALER smokes as many times as possible, stopping if he is at max hp."""
        for _ in range(self.DEALER_items.count(3)):
            if self.DEALER_hp == 4: break
            self.smoke()

    def normalCheat(self):
        """The cheating DEALER Algorithm, it makes the round live, (uses magnifiying glass if it has one), smokes if it can, cuffs AI if it can, then it shoots the AI."""
        self.riggedDetermine(live=True)
        self.magnifier()
        self.DEALERSmoke()
        self.cuff()
        self.DEALERshootAI()
        
    def superCheat(self):
        """The SUPER cheating DEALER Algorithm, it makes the round blank, (uses magnifiying glass if it has one), smokes if it can, then it shoots itself; 
            it makes the round live (uses magnifiying glass if it has one), saws the gun if it can, cuffs the AI if it can, then shoots the AI."""
        self.riggedDetermine(live=False)
        self.magnifier()
        self.DEALERSmoke()
        self.DEALERshootDEALER()
        self.riggedDetermine(live=True)
        self.magnifier()
        self.saw()
        self.cuff()
        self.DEALERshootAI()

    def guessLive(self):
        self.inverter()
        self.DEALERSmoke()
        self.cuff()
        self.saw()
        self.DEALERshootAI()
        self.drinkBeer()

    def guessBlank(self):
        self.drinkBeer()
        self.inverter()
        self.DEALERSmoke()
        self.DEALERshootDEALER()

    def dontCheat(self):
        """The simple algorithm for the DEALER, it randomly guesses if it is live or blank and then plays accordingly."""
        if random.random() < 0.5: self.guessLive()
        else: self.guessBlank()

    def DEALERalgo(self):
        """The DEALER Algorithm used in place of a real dealer, it has to cheat, but it efficiently trains the AI."""
        if self.blank_shells and self.live_shells:
            if random.random() < 0.1 or self.DEALER_hp == 1: self.superCheat()
            elif random.random() < 0.4: self.normalCheat()
            else: self.dontCheat()
        else: self.dontCheat()
        
    def getState(self):
        return np.array([
            self.AI_hp/4, self.DEALER_hp/4,
            self.live_shells/4, self.blank_shells/4,
            self.shell,
            self.current_round_num/8,
            self.is_sawed,
            self.invert_odds,
            *[item/6 for item in self.AI_items], 
            *[item/6 for item in self.DEALER_items]
        ], dtype=np.float16)
        
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, *,std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features, device=device))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features, device=device))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features, device=device))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features, device=device))
        self.register_buffer("weight_epsilon", torch.empty(self.out_features, self.in_features, device=device))
        self.register_buffer("bias_epsilon", torch.empty(self.out_features, device=device))
        self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.weight_mu.size(1) ** 0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / self.weight_mu.size(1) ** 0.5)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / self.bias_mu.size(0) ** 0.5)

    def forward(self, x):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return torch.nn.functional.linear(x, weight, bias)
    
class NLSCDDDQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, *,
                 skip_connections: list = [], activation: nn.Module = nn.ReLU(),
                 use_noisy: bool = True, fully_noisy: bool = False, noise_std_init: float = 0.4):
        """ Noisy Linear Skip-Connected Dueling Double Deep Q Network \n
        ------------- \n
        Base DDDQN (inputs, outputs, hidden_dims) \n
        Optional NLSC (noisy, fully noisy, noise, skip connections) \n
        Misc (activation) """
        super(NLSCDDDQN, self).__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.skip_connections = skip_connections
        self.use_noisy = True if (use_noisy or fully_noisy) else False
        self.hidden_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = NoisyLinear(prev_dim, hidden_dim, std_init=noise_std_init) if fully_noisy else nn.Linear(prev_dim, hidden_dim, device=device)
            self.hidden_layers.append(layer)
            prev_dim = hidden_dim
        
        self.value_fc = NoisyLinear(prev_dim, 1, std_init=noise_std_init) if use_noisy else nn.Linear(prev_dim, 1, device=device)
        self.advantage_fc = NoisyLinear(prev_dim, output_dim, std_init=noise_std_init) if use_noisy else nn.Linear(prev_dim, output_dim, device=device)

        if skip_connections:
            for (from_layer, to_layer) in self.skip_connections:
                if from_layer == 0:
                    projection_layer = (NoisyLinear(input_dim, hidden_dims[to_layer - 1]) if fully_noisy else nn.Linear(input_dim, hidden_dims[to_layer - 1], device=device))
                    self.skip_projections.append(projection_layer)
                else: self.skip_projections.append(None)

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            if self.skip_connections:
                for (from_layer, to_layer) in self.skip_connections:
                    if to_layer == i + 1:
                        if from_layer == 0:
                            projected_input = self.skip_projections[0](outputs[from_layer])
                            x = x + projected_input  # Avoid in-place operation
                        elif outputs[from_layer].shape[1] == x.shape[1]:
                            x = x + outputs[from_layer]  # Avoid in-place operation
                        else: raise ValueError(f"Shape mismatch: cannot add output from layer {from_layer} with shape {outputs[from_layer].shape} to current layer with shape {x.shape}")
            outputs.append(x)

        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values

class DQNAgent:
    def __init__(self, inputs, outputs):
        self.name = "Buck_NLSCDDDQN_v1a.2.1"
        self.inputs, self.outputs = inputs, outputs
        self.batch_size = 3200000
        self.memory = deque(maxlen=100_000)
        self.model = NLSCDDDQN(inputs, outputs, [80, 80, 80], skip_connections=[(0,3)], use_noisy=True).to(device)
        self.target_model = NLSCDDDQN(inputs, outputs, [80, 80, 80], skip_connections=[(0,3)], use_noisy=True).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss().to(device)
        self.steps = 0
        self.updateTargetNetwork()

    def updateTargetNetwork(self): self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad(): q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def replay(self):
        self.steps += 1
        if len(self.memory) < self.batch_size: return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        q_values = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values
            
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def saveModel(self):
        filename = f"{self.name}_{self.steps}.pth"
        if not os.path.exists("models"): os.makedirs("models")
        model_path = os.path.join("models", filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
        }, model_path)

    def loadModel(self):
        filename = f"{self.name}_{self.steps}.pth"
        if not os.path.exists("models"): os.makedirs("models")
        model_path = os.path.join("models", filename)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps = checkpoint['steps']
            print(f"Model loaded from {model_path}")
        else: raise Exception(f"Model not found in {model_path}")
    
class Worker:
    def __init__(self, worker_id: int, experience_queue: Queue, stop_event: Event, 
                 model_update_event: Event, shared_model_state):
        self.worker_id = worker_id
        self.experience_queue = experience_queue
        self.stop_event = stop_event
        self.model_update_event = model_update_event
        self.shared_model_state = shared_model_state
        self.game = Game()
        self.local_agent = DQNAgent(24, 8)
    
    def run(self):
        while not self.stop_event.is_set():
            if self.model_update_event.is_set():
                self.local_agent.model.load_state_dict(self.shared_model_state)
                self.model_update_event.clear()
            
            self.game.resetGame()
            state = self.game.getState()
            done = turn_done = False
            
            if self.game.AI_can_play:
                while not turn_done:
                    action = self.local_agent.act(state)
                    reward = 0
                    
                    match action:
                        case 0: reward = self.game.AIshootDEALER(); turn_done = True
                        case 1: reward = self.game.smoke(player=True)
                        case 2: reward = self.game.magnifier(player=True)
                        case 3: reward = self.game.drinkBeer(player=True)
                        case 4: reward = self.game.inverter(player=True)
                        case 5: reward = self.game.cuff(player=True)
                        case 6: reward = self.game.saw(player=True)
                        case 7: reward = self.game.AIshootAI(); turn_done = True
                    
                    next_state = self.game.getState()
                    self.experience_queue.put((state, action, reward, next_state, done))
                    state = next_state
            else:
                self.game.AI_can_play = True
            
            if self.game.DEALER_can_play:
                self.game.DEALERalgo()
            else:
                self.game.DEALER_can_play = True

def train_parallel(num_processes=4):
    experience_queue = Queue()
    stop_event = Event()
    model_update_event = Event()
    
    # Create shared model state
    agent = DQNAgent(24, 8)
    shared_model_state = agent.model.state_dict()
    
    # Create and start workers
    workers = []
    for i in range(num_processes):
        worker = Worker(i, experience_queue, stop_event, model_update_event, shared_model_state)
        p = mp.Process(target=worker.run)
        workers.append(p)
        p.start()
    
    start_time = time.time()
    last_steps = 0
    
    try:
        while agent.steps < 1_000_000:
            # Collect experiences from workers
            while not experience_queue.empty():
                state, action, reward, next_state, done = experience_queue.get()
                agent.remember(state, action, reward, next_state, done)
            
            # Training
            agent.replay()
            
            if agent.steps % 300 == 0:
                agent.updateTargetNetwork()
                shared_model_state = agent.model.state_dict()
                model_update_event.set()
            
            if agent.steps % 1000 == 0:
                current_sps = agent.steps / (time.time() - start_time)
                print(f"Steps: {agent.steps}, SPS: {current_sps:.2f}")
            
            last_steps = agent.steps
        
        agent.saveModel()
        
    finally:
        stop_event.set()
        for p in workers:
            p.join()

if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn')
    train_parallel()
