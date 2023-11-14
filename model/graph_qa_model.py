# coding: utf-8

# ----------------------------------------------------------------
# Author:   Mouad Hakam (e1002601@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from torch import nn
from transformers import BertModel
from typing import List, Optional, Tuple, Union
from torch_geometric.nn import GATConv, Linear, HGTConv
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelForQuestionAnswering
from transformers.models.bert.modeling_bert import BertPreTrainedModel, QuestionAnsweringModelOutput
from torch.autograd import Variable
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.nn import to_hetero
from torch_geometric.utils import convert
import torch_geometric
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import dgl
# Pytorch Geometric provides three ways for the user to create models on heterogeneous graph data:

# 1) Automatically convert a homogenous GNN model to a heterogeneous GNN model by making use of torch_geometric.nn.to_hetero() or torch_geometric.nn.to_hetero_with_bases().

# 2) Define inidividual functions for different types using PyGs wrapper torch_geometric.nn.conv.HeteroConv for heterogeneous convolution.

# 3) Deploy existing (or write your own) heterogeneous GNN operators.

# More on https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

dicto = {"CD":0,
        "CC"  : 1,
        "DT" : 2,
        "EX" : 3,
        "FW" : 4,
        "IN" : 5,
        "JJ" : 6,
        "JJR" : 7,
        "JJS" : 8,
        "LS" : 9,
        "MD" : 10,
        "NN" : 11,
        "NNS" : 12,
        "NNP" : 13,
        "NNPS" : 14,
        "PDT" : 15,
        "POS" : 16,
        "PRP" : 17,
        "PP$" : 18,
        "RB" : 19,
        "RBR" : 20,
        "RBS" : 21,
        "RP" : 22,
        "SYM" : 23,
        "TO" : 24,
        "UH" : 25,
        "VB" : 26,
        "VBD" : 27,
        "VBG" : 28,
        "VBN"  : 29,
        "VBP" : 30,
        "VBZ" : 31,
        "WDT" : 32,
        "WP" : 33,
        "WP$" : 34,
        "WRB" : 35,
        "#" : 36,
        "." : 37,
        "$" : 38,
        "," : 39,
        ":" : 40,
        "(" : 41,
        ")" : 42,
        "`" : 43,
        "ROOT" : 44,
        "ADJP" : 45, 
        "ADVP" : 46, 
        "NP" : 47, 
        "PP" : 48, 
        "S" : 49, 
        "SBAR" : 50, 
        "SBARQ" : 51, 
        "SINV" :52 , 
        "SQ" :53 , 
        "VP" :54 , 
        "WHADVP" : 55, 
        "WHNP" : 56, 
        "WHPP" :57 , 
        "X" : 58,
        "*" : 59, 
        "0" : 59, 
        "T" : 59, 
        "NIL" :59 , 
        "PRT":60,
        "PRT$": 61,
        "PRP$": 62,
        "HYPH": 63,
        "NML":64,
        "AFX":65,
        "-LRB-":66,
        "CONJP" : 67,
        "-RRB-" : 68,
        "``": 69,
        "INTJ" : 70,
        "QP" : 71,
        "''" : 72,
        "WHADJP" : 73,
        "PRN" : 74,
        "UCP" : 75,
        "ADD" : 76,
        "FRAG" : 77,
        "NFP"  : 78,
        "RRC" : 79,
        'LST' : 80
}
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import random

    
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
# def plot_tree(g,a,b,p):
#     color_map = []
#     node_size = []
#     for node in g:
#         #print(node)
#         if node == a or node == b or node == p :
#             print("yes",node,a,b)
#             color_map.append('blue')
#             node_size.append(20)
#         else: 
#             color_map.append('orange') 
#             node_size.append(10)
#     pos = hierarchy_pos(g) 
#     nx.draw(
#         g,
#         pos,
#         with_labels=False,
#         node_size=node_size,
#         node_color=color_map,
#         arrowsize=4,
#     )
#     plt.show()
#     plt.savefig("gra.jpeg")

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['context'])

class GraphQA(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        
        # Hyper Parameters
        self.num_labels = config.num_labels
        self.node_types = ["token", "leaf", "constituent"]
        self.metadata = (
            # Node
            ["token", "leaf", "constituent"],
            # Edge
            [
                ('token', 'connect', 'token'),
                ('constituent', 'connect', 'constituent'),          
                ('constituent', 'rev_connect', 'constituent'), 
                ('constituent', 'connect', 'token'),     
                ('token', 'rev_connect', 'constituent'),
            ]
        )
        self.graph_hidden_channels = 768
        self.number_of_constituents = 82
        self.input_shape = 768
        self.graph_layer = 2
        self.graph_head = 2

        # BERT Backbone
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = Linear(config.hidden_size, 2)

        # Heterogenous Graph
        self.lin_dict = torch.nn.ModuleDict()

        self.lin_dict["token"] = Linear(config.hidden_size, self.graph_hidden_channels)
        self.lin_dict["constituent"] = Linear(self.number_of_constituents, self.graph_hidden_channels)
        
        
        self.convs = torch.nn.ModuleList()
        for _ in range(self.graph_layer):
            conv = HGTConv(self.input_shape, self.graph_hidden_channels, self.metadata, self.graph_head, group='sum')
            self.convs.append(conv)
        
        self.graph_qa_outputs = Linear(self.graph_hidden_channels, 1)
        self.graph2_qa_outputs = Linear(self.graph_hidden_channels, 2)

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        sep_index: Optional[torch.Tensor] = None,
        graph_data: Optional[dict] = None,
        return_dict: Optional[bool] = None,
        test: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        
        input_ids.to("cuda:0")
        attention_mask.to("cuda:0")
        token_type_ids.to("cuda:0")
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        data = HeteroData()
        data["constituent"].node_id = graph_data[0]["graph"].to_dict()["constituent"]["node_id"]
        data["constituent"].x = graph_data[0]["graph"].to_dict()["constituent"]["x"]
        data["constituent"].y = graph_data[0]["graph"].to_dict()["constituent"]["y"]
        data["token"].node_id = torch.arange(outputs[0].size(1))
        data["token"].x = outputs[0][0]
        transform = T.Compose([ T.AddSelfLoops(),T.ToUndirected()])
        transform2 = T.AddSelfLoops()
        transform1 = T.RemoveIsolatedNodes()

        data["constituent","connect","constituent"].edge_index =  graph_data[0]["graph"].to_dict()["_global_store"]["('constituent', 'connect', 'constituent')"].t()
        data["constituent","connect","token"].edge_index =  graph_data[0]["graph"].to_dict()["_global_store"]["('constituent', 'connect', 'token')"].t()
        data.to("cpu")
        
        datai = transform1(data)
        g = convert.to_dgl(datai)

        datai = datai.to("cpu")
        data2 = datai.to_homogeneous()
        c = convert.to_dgl(data2)

        
        
        y = graph_data[0]["graph"].to_dict()["constituent"]["y"].float()
        
        kop = y
        lp_type = None
        if torch.argmax(y).item() != 0 and end_positions.item() != 0:
            mop = []
            
            lp = torch.argmax(data["constituent"].x[torch.argmax(y).item()])
            for key, value in dicto.items():
                if lp.item() == value:
                    lp_type = key
            for ter in data["constituent"].x:
                if torch.equal(ter,data["constituent"].x[torch.argmax(y).item()]):
                    mop.append(1)
                else:
                    mop.append(0)
            kop = torch.tensor(mop)

            result = torch.cat(dgl.traversal.bfs_nodes_generator(c,torch.argmax(y).item()))
            msk = result.ge(g.num_nodes("constituent"))
            tok = torch.masked_select(result, msk)
            lm = outputs[0].size(1) -1
            l_start = torch.zeros(outputs[0].size(1))
            l_end = torch.zeros(outputs[0].size(1))
            num_nodes = c.num_nodes()
            mini = torch.min(tok).item()
            maxi = torch.max(tok).item()
            e_ind = min(num_nodes - mini,lm)  
            s_ind = min(num_nodes -  maxi,lm) 
            if s_ind == start_positions.item() or e_ind == end_positions.item() :
                l_start[s_ind] = 5
                l_end[e_ind] = 5
            
        data = transform(data)

        data.to("cuda:0")

        tokens = graph_data[0]["graph"].to_dict()["token"]["x"].squeeze()[torch.nonzero(graph_data[0]["graph"].to_dict()["token"]["x"].squeeze())].squeeze()
        positions = graph_data[0]["graph"].to_dict()["constituent"]["y"].float().to("cuda:0")

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict


        for node_type, x in x_dict.items():

            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)  

        logits = self.graph_qa_outputs(x_dict['constituent'])
        logits2 = self.graph2_qa_outputs(x_dict['token'])
        start_logits, end_logits = logits2.split(1, dim=-1)
        logits.to("cuda:0")
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits   = end_logits.squeeze(-1).contiguous()
        logits = logits.squeeze(1)
        l_start = torch.zeros(outputs[0].size(1))
        l_end = torch.zeros(outputs[0].size(1))
        
        kop = kop.to("cuda:0")
        y =y.to("cuda:0")
        
        m = torch.argmax(logits).item()
        if  m!=0 :
            result = torch.cat(dgl.traversal.bfs_nodes_generator(c, m))
            msk = result.ge(g.num_nodes("constituent"))
            tok = torch.masked_select(result, msk)
            lm = outputs[0].size(1)-1
            e_index = min(c.num_nodes() - torch.min(tok).item(), lm)
            s_index = min(c.num_nodes() - torch.max(tok).item(), lm)
            l_start[s_index] = torch.max(logits).item()
            l_end[e_index] = torch.max(logits).item()

        l_end = l_end.to("cuda:0")
        l_start = l_start.to("cuda:0")
        start_logits = start_logits.to("cuda:0")
        end_logits = end_logits.to("cuda:0")
        s_logits  = start_logits.add(l_start)
        e_logits  = end_logits.add(l_end)
        y = y.to("cuda:0")
        total_loss = None

        start_positions = start_positions.to("cuda:0")
        end_positions = end_positions.to("cuda:0")

        if start_positions is not None and end_positions is not None and test == False:
            loss_fct = CrossEntropyLoss()
            s_logits = s_logits.unsqueeze(0)
            e_logits = e_logits.unsqueeze(0)
            start_loss = loss_fct(s_logits,start_positions)
            end_loss = loss_fct(e_logits,end_positions)
            loss = loss_fct(logits,y)
            total_loss = (start_loss + end_loss + loss)/3

        if not return_dict:
            output = (s_logits, e_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else [output,lp_type]

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=s_logits,
            end_logits=e_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    print("made it")