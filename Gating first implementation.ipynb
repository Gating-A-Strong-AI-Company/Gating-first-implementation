# %%
import torch
import torch.nn as nn
import math

# %% [markdown]
# # Each gate is a class

# %%
class Gate():
    """
    Implements gates for transient rewiring of a network. This is inspired by the paper:
    
    Nikolić, D. (2023). Where is the mind within the brain? Transient selection of subnetworks 
    by metabotropic receptors and G protein-gated ion channels. 
    Computational Biology and Chemistry, 103, 107820.

    Note: currently works only for 1D layers of neurons
    
    x, y: determine which connection is being gated by this gate, where x and y are indexes of neuron in
        input and output layer for that weight, respectively.
        x can have the value "bias" in case the gate gates the bias parameter.
        
    trigger_layer: in which layer the trigger for this gate is located, input or output. The values
            are "down" for input and "up" for output.
            
    trigger_neuron_id: the neuron index that will provide inputs for triggering this gate
    
    g_activated: the gating value g that will be set once the gate is activated. 
    
    activation_threshold: The minimal output value of the trigger neuron necessary to activate the gates. 
    The default value is 0.1. 
    
    g_default: The value of the gating parameter in its inactivatged state. The deafult value is 1.0.
    
    duration: deterimines the number of iteratations after which the gate will return back 
        to its default state. The default number of iterations is 5.
    
    Usage:
    gate = Gate(g, neuron_id)
    
    
    We have the following methods:
    
    
    get_trigger_neuron(): returns the trigger neuron index and the layer (up or down)
    
    sniff(): check the input and decides whether to activate the gate; it does nothing if the gates is
            already active.
            
    
    
    """

    def __init__(self, x, y, trigger_layer, trigger_neuron_id, g_activated, activation_threshold = 0.1, g_default = 1.0, duration = 5):

        if x == "bias":
            self.n_x = -1
        else:
            self.n_x = x
        self.n_y = y
        if trigger_layer == "down":
            self.layer = trigger_layer #up or down
        else:
            self.layer = "up"
        self.g_activated = g_activated  #torch.tensor(g_activated)
        self.neuron_id = trigger_neuron_id
        self.activation_threshold = activation_threshold
        self.g_default = g_default
        self.duration = duration
        self.state = self.g_default
        self.counter = 0
        self.activated = False

    def get_trigger_neuron(self):
        return self.layer, self.neuron_id

    def sniff(self, input):
        if self.counter == 0 and input > self.activation_threshold:
                self.state = self.g_activated
                self.counter = self.duration
                self.activated = True
        self.counter_()

    def counter_(self):
        if self.counter > 0:
            self.counter -= 1
        else:
            self.state = self.g_default
            self.activated =  False
            
    def de_activate(self):
        self.activated = False
        self.counter = 0
        self.state = self.g_default


# %% [markdown]
# # Here we expand the functionality of a PyTorch layer to accomodate gates

# %%


class GatedLinear(nn.Module):
    """
    This overwrites the Linear function of the nn module. The new forward function also applies gates
    
    The constructor receivies a list of gates to operate within this layer
    
    forward() is extended so that it respects the gate values
    
    think(): this is a function that updates gates of that layer
    
    
    """
    def __init__(self, in_features, out_features, set_gates=None, bias=True):
        super(GatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if set_gates is not None:
            self.gating = True
            self.g_weight = nn.Parameter(torch.ones(out_features, in_features))
            self.g_weight.requires_grad=False
            if bias:
                self.g_bias = nn.Parameter(torch.ones(out_features))
                self.g_bias.requires_grad=False
            else:
                self.register_parameter('g_bias', None)
            
        else:
            self.gating = False
            self.register_parameter('g_weight', None)
            self.register_parameter('g_bias', None)
            
        # Initialize the weights and bias
        self.reset_parameters()        
        if set_gates is not None:        
            self.gates = set_gates()  #can we add this one with nn.Parameter? This should all be torch tensors      
        

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, use_gates = True):            
        if self.gating is True and use_gates is True:
            linear_output = torch.matmul(input, self.weight.t() * self.g_weight.t()) 
        else:
            linear_output = torch.matmul(input, self.weight.t())
            
        if self.bias is not None:
            if self.gating is True and use_gates is True:
                linear_output += self.g_bias * self.bias
            else:
                linear_output += self.bias
            
        return linear_output
        
    
    def think(self, output_down, output_up):
        #initialize gates to ones; dimensionality outputs_1 x outputs_2
        self.g_weight.fill_(1.0)
        self.g_bias.fill_(1.0)
        
        for g in self.gates:
            layer, n_id = g.get_trigger_neuron()
            if layer == "down":
                g.sniff(output_down[n_id])
                #print("debug",n_id, output_down[n_id])
            else:
                g.sniff(output_up[n_id])
            
            if g.n_x >= 0:
                self.g_weight[g.n_x, g.n_y] *= g.state
            else:
                self.g_bias[g.n_y] *= g.state
                
        #print("g weight", self.g_weight)
        return self.gates
    
    def de_activate_all_gates(self):
        for g in self.gates:
            g.de_activate()
        
    

# %% [markdown]
# # Creating a neural network layer with gates

# %%
#Manually create and array of gates
def my_gates():
    g1 = Gate(0, 1, "down", trigger_neuron_id = 0, g_activated = 10)
    g2 = Gate(1, 0, "down", trigger_neuron_id = 1, g_activated = 10)
    g3 = Gate('bias', 1, "down", trigger_neuron_id = 0, g_activated = 0.1)
    return [g1, g2, g3]

#Create a single neural network layer and pass to it the gates
gl = GatedLinear(2,2, set_gates = my_gates)

#Set the connections and biases for the layer
gl.weight = nn.Parameter(torch.tensor([[ 0.5, 0.5],
                                        [0.5, 0.5]]))
gl.bias = nn.Parameter(torch.tensor([0.0, -.5]))
#print(gl.weight, gl.bias)

# %% [markdown]
# # Create a sequence of inputs for the layer to iterate through

# %%
inps = [torch.tensor([0.0,0.0]), 
        torch.tensor([0.0,1.0]), 
        torch.tensor([1.0,0.0]), 
        torch.tensor([1.0,1.0])]

# %% [markdown]
# # Finally, let the layer of neurons iterated through the inputs using gates

# %%
for inp in inps:
    out = gl.forward(inp, use_gates = True)
    gl.think(inp, out)
    print(out)

gl.de_activate_all_gates()


# %% [markdown]
# # Compare to the outputs without using gates

# %%
for inp in inps:
    out = gl.forward(inp, use_gates = False)
    print(out)

# %% [markdown]
# # Let us now build a PyTorch model

# %% [markdown]
# ## First we need a set of gates

# %%
def linear1_gates():
    g1 = Gate(0, 1, "down", trigger_neuron_id = 0, g_activated = 0.2)
    g2 = Gate(1, 0, "down", trigger_neuron_id = 1, g_activated = 0.1)
    g3 = Gate('bias', 1, "down", trigger_neuron_id = 0, g_activated = 0.1)
    return [g1, g2, g3]

# %% [markdown]
# ## Next, we define a model

# %%
class GatedModel(torch.nn.Module):

    def __init__(self):
        super(GatedModel, self).__init__()

        self.linear1 = GatedLinear(5, 4, set_gates = linear1_gates)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4, 3)
        self.softmax = torch.nn.Softmax(dim=0)
    

    def forward(self, input_x, use_gates = True):
        x = self.linear1(input_x, use_gates=use_gates) 
        x_L1 = self.activation(x)
        self.linear1.think(input_x, x_L1)
        
        x = self.linear2(x_L1)
        x_L2 = self.softmax(x)
        
        return x_L2
    
    def iterate (self, input_data, use_gates=True, n=2):
        for it in range(n):
            output = self.forward(input_data, use_gates=use_gates)
            print(output)
            

# %% [markdown]
# ## A single data point that will be "looked" at multiple times

# %%
input_data = torch.tensor([1., 1., 1., 1., 2.])

# %% [markdown]
# ## Create a gating model

# %%
my_first_gated_model = GatedModel()
print(my_first_gated_model)

# %% [markdown]
# ## Look three times into the same input

# %%
my_first_gated_model.iterate(input_data, use_gates = True, n = 3)
#The outputs should change from first to second look and then stabilize
#This demonstrates the contribution of gates

# %% [markdown]
# ## The same as above but without using gates

# %%
my_first_gated_model = GatedModel()
my_first_gated_model.iterate(input_data, use_gates = False, n = 3)
#The outputs do not change with repeated looks

# %%


# %%


# %%


# %%



