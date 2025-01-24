import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using: {device}")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, *,std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias_mu = nn.Parameter(torch.empty(out_features, device=device))
        self.bias_sigma = nn.Parameter(torch.empty(out_features, device=device))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features, device=device))
        self.register_buffer("bias_epsilon", torch.empty(out_features, device=device))
        self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / self.weight_mu.size(1)**0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / self.weight_mu.size(1)**0.5)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / self.bias_mu.size(1)**0.5)

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
        """
        Noisy Linear Skip-Connected Dueling Double Deep Q Network \n
        ------------- \n
        Base DDDQN (inputs, outputs, hidden_dims) \n
        Optional NLSC (noisy, fully noisy, noise, skip connections) \n
        Misc (activation) \n
        ------------- \n
        """
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
                else:
                    self.skip_projections.append(None)

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            if self.skip_connections:
                for (from_layer, to_layer) in self.skip_connections:
                    if to_layer == i + 1:
                        if from_layer == 0:
                            projected_input = self.skip_projections[0](outputs[from_layer])
                            x += projected_input
                        elif outputs[from_layer].shape[1] == x.shape[1]:
                            x += outputs[from_layer]
                        else: raise ValueError(f"Shape mismatch: cannot add output from layer {from_layer} with shape {outputs[from_layer].shape} to current layer with shape {x.shape}")
                outputs.append(x)

        value = self.value_fc(x).expand(x.size(0), self.advantage_fc.out_features)
        advantage = self.advantage_fc(x) - self.advantage_fc(x).mean(dim=1, keepdim=True)
        q_values = value + advantage
        return q_values

class DQNAgent:
    def __init__(self, inputs, outputs):
        self.name = "Buck_NLSCDDDQN_v0.4.4"
        self.steps = 0
        self.inputs = inputs
        self.outputs = outputs
        self.memory_size = 150_000
        self.batch_size = 256
        self.memory = deque(maxlen=self.memory_size)
        self.model = NLSCDDDQN(inputs, outputs, [80, 80, 80], skip_connections=[(0,3)], use_noisy=True).to(device)
        self.target_model = NLSCDDDQN(inputs, outputs, [80, 80, 80], skip_connections=[(0,3)], use_noisy=True).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0006)
        self.loss_fn = nn.MSELoss().to(device)
        self.updateTargetNetwork()

    def updateTargetNetwork(self): self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad(): q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return #FIX ?

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states, next_states = np.array(states), np.array(next_states)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        q_values = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values
            
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def saveModel(self, filename):
    filename = f"{self.name}_{self.steps}.pth"
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model_path = os.path.join("models", filename)
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'self.steps': agent.steps,
    }, model_path)

def loadModel(self, filename):
    filename = f"{self.name}_{self.steps}.pth"
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model_path = os.path.join("models", filename)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.steps = checkpoint['self.steps']
        print(f"Model loaded from {model_path}")
    else: raise Exception(f"Model not found in {model_path}")

running = True
class Game():
    def __init__(self):
        """Initializes the game state and shotgun."""
        self.max_shells = 8
        self.live_shells, self.blank_shells, self.shells, self.shell, self.current_round_num, self.round = 0
        self.AI_items, self.DEALER_items = [] # 0:nothing, 1:beer 2:magnifier 3:smoke 4:inverter 5:cuffs 6:saw
        self.AI_can_play, self.DEALER_can_play = True
        self.AI_hp, self.DEALER_hp = 4
        self.invert_odds, self.is_sawed = False
        self.resetShells()
    
    def resetShells(self):
        """Adds a random number of live and blank shells to the shotgun."""
        self.live_shells, self.blank_shells = random.randint(1, self.max_shells//2), random.randint(1, self.max_shells//2)
        self.shells = self.totalShells()
        self.current_round_num = 0
    
    def totalShells(self): return self.live_shells + self.blank_shells
    
    def determineShell(self):
        """Determines the type of shell in the current chamber, returns 1 for live and 0.5 for blank."""
        return 1 if random.random() <= (self.live_shells/self.shells) else 0.5
    
    def riggedDetermine(self, live: bool):
        """Determines the shell to be the chosen shell."""
        return 1 if live else 0.5
    
    def removeShell(self, live: bool):
        if live: self.live_shells -= 1
        else: self.blank_shells -= 1
    
    def resetGame(self):
        """Resets the game state, initializes the shotgun, and loads bullets."""
        self.resetShells()
        self.AI_items = []
        self.DEALER_items = []
        self.AI_hp = 4
        self.DEALER_hp = 4
    
    def debugPrintGame(self):
        """Prints the current game state for debugging and visualization."""
        print(f"Current Round: {self.round}")
        print(f"AI HP: {self.AI_hp}, DEALER HP: {self.DEALER_hp}")
        print(f"Live Shells: {self.live_shells}, Blank Shells: {self.blank_shells}")
        print(f"Shells in Shotgun: {self.shells}")
        print(f"AI Items: {self.AI_items}")
        print(f"DEALER Items: {self.DEALER_items}")
        print(f"Current Round Number: {self.current_round_num}")
        print(f"Is sawed?: {self.is_sawed}")
        print(f"Invert Odds?: {self.invert_odds}")
        #WIP
    
    def restockItems(self):
        """Restocks the round for the AI and DEALER."""
        for _ in range(self.round*2):
            if len(self.AI_items) < 8:
                self.AI_items.append(random.randint(1, 6)) 
                print("AI round: ", self.AI_items)
            if len(self.DEALER_items) < 8:
                self.DEALER_items.append(random.randint(1, 6)) 
                print("DEALER round: ", self.DEALER_items)
    
    def removeUnknownShell(self):
            if random.randint(0, 1) == 1 and self.live_shells > 0:
                self.live_shells -= 1
            else: self.blank_shells -= 1
    def drinkBeer(self, player: bool = False):
        """Player drinks beer, returns reward."""
        if self.totalShells == 1:
            raise Exception("are wii gunna have a problem?") 
            #WIP
        if player:
            if 1 in self.AI_items:
                self.AI_items.remove(1)
                if self.shell == 0: self.removeUnknownShell()
                elif self.shell == 1: self.live_shells -= 1
                else: self.blank_shells -= 1
                return 0.5
            else: return -1
        else:
            if 1 in self.DEALER_items:
                self.DEALER_items.remove(1)
            
    def magnifier(self, player: bool = False):
        """Player breaks glass, determining shell, returns reward."""
        if player:
            if 2 in self.AI_items:
                self.AI_items.remove(2)
                self.shell = self.determineShell()
                return 1
            else: return -1
        else:
            if 2 in self.DEALER_items:
                self.DEALER_items.remove(2)           
    
    def smoke(self, player: bool = False):
        """Player smokes, regains 1 hp, returns reward."""
        if player:
            if 3 in self.AI_items:
                self.AI_items.remove(3)
                self.AI_hp = min(4, self.AI_hp+1)
                return 1
            else: return -1
        else:
            if 3 in self.DEALER_items:
                self.DEALER_items.remove(3)
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
                if self.shell == 0.5: self.blank_shells -= 1; self.live_shells += 1
                elif self.shell == 1: self.blank_shells += 1; self.live_shells -= 1
                self.invert()
                return 0.3
            else: return -1
        else:
            if 4 in self.DEALER_items:
                self.DEALER_items.remove(4)
    
    def cuff(self, player: bool = False):
        """Player cuffs opponent, skipping their turn, returns reward."""
        if player:
            if 5 in self.AI_items:
                self.AI_items.remove(5)
                self.DEALER_can_play = False
                return 1
            else: return -1
        else:
            if 5 in self.DEALER_items:
                self.DEALER_items.remove(5)
                self.AI_can_play = False
    
    def saw(self, player: bool = False):
        """Player saws off shotgun, doubling damage, returns reward."""
        if player:
            if 6 in self.AI_items:
                self.AI_items.remove(6)
                self.is_sawed = True
                return 1 if self.shell != 0.5 else -2
            else: return -1
        else:
            if 5 in self.DEALER_items:
                self.DEALER_items.remove(5)
                self.is_sawed = True
    
    def AIshootAI(self):
        """Determines the outcome of the shot if not already known, and shoots AI, returns reward."""
        if self.shell == 0:
            self.shell = self.determineShell()
            if self.shell == 1:
                self.AI_hp -= 1 if self.is_sawed == False else 2
                return -3 if self.is_sawed == False else -6 
            else: return 0 if self.is_sawed == False else -2
        elif self.shell == 0.5: return 2 if self.is_sawed == False else -8
        else:
            if self.is_sawed == False: self.AI_hp -= 1; return -20
            else: self.is_sawed = False; self.AI_hp -= 2; return -40
        
    def AIshootDEALER(self, shell):
        """Determines the outcome of the shot if not already known, and shoots DEALER, returns reward."""
        if shell == 0:
            shell = self.determineShell()
            if shell == 1:
                if self.is_sawed == False: self.DEALER_hp -= 1; return 3
                else: self.is_sawed = False; self.DEALER_hp -= 2; return 6
            else: return 0
        elif shell == 1:
            if self.is_sawed == False: self.DEALER_hp -= 1; return 4
            else: self.is_sawed = False; self.DEALER_hp -= 2; return 8
        else:
            if self.is_sawed == False: return -20 
            else: self.is_sawed = False; return -32
    
    def DEALERshootDEALER(self):
        """Determines the outcome of the shot if not already known, and shoots DEALER."""
        if shell == 0.5: self.AI_can_play = False
        elif shell == 0:
            shell = self.determineShell()
            if shell == 1:
                self.DEALER_hp -= 1
    
    def DEALERshootAI(self):
        """Determines the outcome of the shot if not already known, and shoots AI."""
        if shell == 1: self.AI_hp -= 1 if self.is_sawed == False else 2
        else:
            shell = self.determineShell()
            if shell == 1: self.AI_hp -= 1 if self.is_sawed == False else 2
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
        """The simple algorithm for the DEALER, it randomly guesses if it is live or blank and then plays accordingly"""
        if random.random() < 0.5: self.guessLive()
        else: self.guessBlank()

    def DEALERalgo(self):
        """The DEALER Algorithm used in place of a real dealer, it has to cheat, but it efficiently trains the AI"""
        shells = self.blank_shells and self.live_shells
        canSuperCheat = (random.random() < 0.1) and shells
        canCheat = (random.random() < 0.3) and shells and not canSuperCheat
        cantCheat = not canCheat and not canSuperCheat
        if cantCheat: self.dontCheat()
        elif canCheat: self.normalCheat()
        else: self.superCheat()
    
def playGame(agent: DQNAgent, game: Game):
    game.resetGame()
    def getState():
        flattened_state = np.array(#WIP
            dtype=np.float32)
        print(flattened_state)
        return flattened_state
    
    state = getState()
    done = False
    while not done:
        while game.AI_can_play and not done:
            action = agent.act(state)
            reward = 0
            if action == 0: reward = game.AIshootDEALER(game.shell); done = True
            elif action == 1: reward = game.smoke(player=True)
            elif action == 2: reward = game.magnifier(player=True)
            elif action == 3: reward = game.drinkBeer(player=True)
            elif action == 4: reward = game.inverter(player=True)
            elif action == 5: reward = game.cuff(player=True)
            elif action == 6: reward = game.saw(player=True)
            elif action == 7: reward = game.AIshootAI(game.shell); done = True

            next_state = getState()
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay()
            if agent.steps % 10 == 0:
                agent.updateTargetNetwork()

        if game.DEALER_can_play:
            game.DEALERalgo()

        game.AI_can_play = True
        game.DEALER_can_play = True

agent = DQNAgent(22, 8); lastSteps, e = 0
while True:
    e += 1
    if (e) % 10 == 0:
        _steps = agent.steps
        for ep in range(20):
            playGame(agent, train=False)

        print(f"{(agent.steps - lastSteps) // 20}")
        agent.steps = _steps
    else: playGame(agent)

    if agent.steps > 1_000_000: saveModel(agent); break
    lastSteps = agent.steps
