import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal

class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()
        

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False,missions=False):
        if missions is not False :
            value, x, states = self(inputs, states, masks,missions)
        else:
            value, x, states = self(inputs, states, masks)
            
            
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions,missions=False):
        if missions is not False :
            value, x, states = self(inputs, states, masks,missions)
        else:
            value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class RecMLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(RecMLPPolicy, self).__init__()

        self.action_space = action_space
        assert action_space.__class__.__name__ == "Discrete"
        num_outputs = action_space.n

        self.p_fc1 = nn.Linear(num_inputs, 64)
        self.p_fc2 = nn.Linear(64, 64)
        self.p_fc3 = nn.Linear(64, 32)
        
        self.conv1_3= nn.Conv2d(3,32,3)
        self.conv2_3= nn.Conv2d(32,32,3)
        self.conv3_3= nn.Conv2d(32,32,3)
        
        self.conv1_2= nn.Conv2d(3,4,2)
        self.conv2_2= nn.Conv2d(4,8,2)
        self.conv3_2= nn.Conv2d(8,16,2)
        self.conv4_2= nn.Conv2d(16,32,2)
        self.conv5_2= nn.Conv2d(32,32,2)
        self.conv6_2= nn.Conv2d(32,32,2)
        
        self.conv1_4= nn.Conv2d(3,32,4)
        self.conv2_4= nn.Conv2d(32,32,4)
        
         # models used to mix text and image inputs
        self.adapt_1 = nn.Linear(7,num_inputs)
        self.adapt_2 = nn.Linear(num_inputs,64)
        self.adapt_3 = nn.Linear(2*64,64)



        self.v_fc1 = nn.Linear(64, 64)
        self.v_fc2 = nn.Linear(64, 32)
        self.v_fc3 = nn.Linear(32, 1)
        
        self.a_fc1 = nn.Linear(64, 64)
        self.a_fc2 = nn.Linear(64, 64)
        #self.a_fc3 = nn.Linear(32, action_space.n)
        
        self.preGru = nn.Linear(128, 64)


        # Input size, hidden size
        self.gru = nn.GRUCell(64, 64)
        
        self.postGru = nn.Linear(64, 64)

        self.dist = Categorical(64, num_outputs)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        """
        Size of the recurrent state of the model (propagated between steps
        """
        return 64

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        orthogonal(self.gru.weight_ih.data)
        orthogonal(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks, missions=False):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputForMLP = inputs.view(-1, batch_numel)
        
        c0 = F.relu(self.p_fc1(inputForMLP))
        c0 = F.relu(self.p_fc2(c0))
        c0 = F.relu(self.p_fc3(c0))
        
        
        c2 = F.relu(self.conv1_2(inputs))
        c2 = F.relu(self.conv2_2(c2))
        c2 = F.relu(self.conv3_2(c2))
        c2 = F.relu(self.conv4_2(c2))
        c2 = F.relu(self.conv5_2(c2))
        c2 = F.relu(self.conv6_2(c2))
        c2=c2.view(-1,32)
        
        
        c3 = F.relu(self.conv1_3(inputs))
        c3 = F.relu(self.conv2_3(c3))
        c3 = F.relu(self.conv3_3(c3))
        c3=c3.view(-1,32)
        
        c4 = F.relu(self.conv1_4(inputs))
        c4 = F.relu(self.conv2_4(c4))
        c4=c4.view(-1,32)
        
        #print('c2', c2.size())
        #print('c3', c3.size())
        #print('c4', c4.size())
        
        x=torch.cat([c0,c2, c3, c4],dim=1)

        #print('concat', x.size())
        #x= F.relu(self.adapt_3(missions))

       
        
#        if missions is not False:
#            missions=F.relu(self.adapt_1(missions))
#            missions=F.relu(self.adapt_2(missions))
#            missions=torch.cat([missions, x],dim=1)
#            x= F.relu(self.adapt_3(missions))

        x=F.relu(self.preGru(x))

        assert inputs.size(0) == states.size(0)
        x=self.gru(x, states * masks)
        x = states = self.postGru(x)
        
        #x = states = self.gru(x, states * masks)
        
        actions = self.a_fc1(x)
        actions = F.relu(actions)
        actions = self.a_fc2(actions)
        actions = F.relu(actions)
        #actions = self.v_fc3(actions)

        value = self.v_fc1(x)
        value = F.relu(value)
        value = self.v_fc2(value)
        value = F.relu(value)
        value = self.v_fc3(value)
        value = F.relu(value)

        return value, actions, states

class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)
        
        self.adapt_1 = nn.Linear(4096,num_inputs)
        self.adapt_2 = nn.Linear(2*num_inputs,num_inputs)
        

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks,missions=False):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)
        
        
        if missions is not False:
            missions=self.adapt_1(missions)
            missions=torch.cat([missions, inputs],dim=1)
            inputs=self.adapt_2(missions)
            

        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x, states
    

class easyPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(easyPolicy, self).__init__()

        self.action_space = action_space

        
        self.adapt_1 = nn.Linear(7,10)
        self.adapt_2 = nn.Linear(10,30)
        self.adapt_3 = nn.Linear(30,64)


        self.adapt_4 = nn.Linear(7,5)
        self.adapt_5 = nn.Linear(5,3)
        self.adapt_6 = nn.Linear(3,1)


        

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks,missions=False):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)
        
        
        x=F.relu(self.adapt_1(missions))
        x=F.relu(self.adapt_2(x))
        x=F.relu(self.adapt_3(x))
        
        
        value=F.relu(self.adapt_4(missions))
        value=F.relu(self.adapt_5(value))
        value=F.relu(self.adapt_6(value))

       

        return value, x, states

def weights_init_cnn(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, use_gru):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        if use_gru:
            self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init_cnn)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        if hasattr(self, 'gru'):
            x = states = self.gru(x, states * masks)

        return self.critic_linear(x), x, states
