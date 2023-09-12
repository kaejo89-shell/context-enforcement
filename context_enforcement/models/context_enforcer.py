import numpy as np
import torch
from torch.nn import MultiheadAttention, Linear, GELU
from torch.nn import Module
from transformers.models.bart.modeling_bart import BartConfig, BartAttention
from transformers.activations import ACT2FN

from torch import nn
from typing import Optional, List


def compute_context_boundary(seq_max_len,
                             context_sampling_bounds=(0.1, 0.45),
                             context_max_len=100):
    """
    Returns the context boundary as a function of the input sequence length, the desired context length

    :param seq_max_len:
    :param context_sampling_bounds:
    :param context_max_len:
    :return:
    """
    boundary = np.random.uniform(context_sampling_bounds[0],
                                 context_sampling_bounds[1])
    boundary_start = int(seq_max_len * boundary)
    boundary_end = boundary_start + context_max_len
    
    if context_max_len >=seq_max_len:
        boundary_start = int(0.15*seq_max_len)
        context_max_len = int(0.70*seq_max_len)
        boundary_end = boundary_start + context_max_len
        
    
    return boundary_start, boundary_end


def split_contexts_with_boundary(hidden_rep,
                                 context_boundary, ):

    boundary_start, boundary_end = context_boundary

    # Separate the representations according to boundaries specified
    left_context = hidden_rep[:, :boundary_start]
    right_context = hidden_rep[:, boundary_end:]
    focus_context = hidden_rep[:, boundary_start:boundary_end]

    return [left_context, focus_context, right_context]

def split_contexts_with_attention(attention_mask, context_boundary, ):

    boundary_start, boundary_end = context_boundary

    # Separate the representations according to boundaries specified
    left_context = attention_mask[:, :boundary_start]
    right_context = attention_mask[:, boundary_end:]
    focus_context = attention_mask[:, boundary_start:boundary_end]

    return [left_context, focus_context, right_context]

class BartContextEnforcement(Module):
    def __init__(self,
                 embed_dim,
                 dropout=0.1,
                 share_mha=False,
                 num_heads=1):
        super().__init__()
        self._share_mha = share_mha
        self._dropout = dropout
        if self._share_mha:
            mha =  BartAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
            self._left_mha = mha
            self._right_mha = mha
        else: 
            self._left_mha =  BartAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
            self._right_mha =  BartAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
            
        self._wlc = Linear(embed_dim, embed_dim)
        self._wrc = Linear(embed_dim, embed_dim)
        self._activate_func = GELU()
    
    def forward(self,
                hidden_rep,
                context_boundary: tuple,
                
                attention_mask: torch.FloatTensor,
            layer_head_mask: torch.FloatTensor,
            output_attentions: Optional[bool] = False,
                 **kwargs):
        """
        Select the context focus and perform the context enforcement

        :param hidden_rep:
        :param context_boundary:
        :param return_attentions:
        :return:
        """

        [left_context, focus_context, right_context] = split_contexts_with_boundary(hidden_rep,
                                                                                    context_boundary, )

        # Compute the context focus attention
        left_focus, left_focus_attention,_ = self._left_mha(focus_context,
                                                          left_context,
                                                          layer_head_mask=layer_head_mask,
                                                          output_attentions=output_attentions, )
        right_focus, right_focus_attention,_ = self._right_mha(focus_context,
                                                             right_context,
                                                             layer_head_mask=layer_head_mask,
                                                          output_attentions=output_attentions, 
                                                             )
        rl_reps = self._activate_func(self._wrc(right_focus) + self._wlc(left_focus))
        focus_combined_rep = rl_reps + focus_context

        # Stitch the full reps together
        full_rep = torch.concat([left_context, focus_combined_rep, right_context], dim=1)
        if not output_attentions:
            return [focus_combined_rep,
                    full_rep], None
        else:
            return [focus_combined_rep, full_rep], [left_focus_attention, right_focus_attention]



class ContextEnforcementCrossed(Module):
    def __init__(self,
                 dim,
                 activation_function,
                 dropout=0.1,
                 share_mha=False,
                 num_heads=1,
                 
                 ):
        super().__init__()
        self._share_mha = share_mha
        self._dropout = dropout
        
        if self._share_mha:
            mha = MultiheadAttention(dim,
                                     num_heads,
                                     dropout=dropout,
                                     batch_first=True)
            self._left_mha = mha
            self._right_mha = mha
        else:
            self._left_mha = MultiheadAttention(dim,
                                                num_heads,
                                                dropout=dropout,
                                                batch_first=True)
            self._right_mha = MultiheadAttention(dim,
                                                 num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        
        self._context_mha = MultiheadAttention(dim,
                                                 num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        
        self._wlc = Linear(dim, dim)
        self._wrc = Linear(dim, dim)
        self._activate_func = ACT2FN[activation_function]
        self._wfc = Linear(dim,dim)

    def forward(self,
                hidden_rep,
                context_boundary: tuple,
                return_attentions=False):
        """
        Select the context focus and perform the context enforcement

        :param hidden_rep:
        :param context_boundary:
        :param return_attentions:
        :return:
        """

        [left_context, focus_context, right_context] = split_contexts_with_boundary(hidden_rep,
                                                                                    context_boundary, )

        # Compute the context focus attention
        left_focus, left_focus_attention = self._left_mha(
                                                          left_context,
                                                          focus_context,
                                                          focus_context )
        right_focus, right_focus_attention = self._right_mha(right_context,
                                                             focus_context,
                                                             focus_context, )
        right_focus = self._activate_func(self._wrc(right_focus) ) + right_context
        left_focus = self._activate_func(self._wlc(left_focus)) + left_context
        
        boundary = torch.zeros((left_focus.shape[0],1,left_focus.shape[-1]),device=left_focus.device)
        context = torch.concat([left_focus, boundary, right_focus], dim=1)
        
        focus_rep,focus_attention = self._context_mha(focus_context,
                                                      context,
                                                      context) 
        focus_rep = self._activate_func(self._wfc(focus_rep)) + focus_context

        # Stitch the full reps together
        full_rep = torch.concat([left_focus,focus_rep, right_focus], dim=1)
        if not return_attentions:
            return [focus_rep,
                    full_rep], None
        else:
            return [focus_rep, full_rep], [left_focus_attention, right_focus_attention]



class ContextEnforcement(Module):
    def __init__(self,
                 dim,
                 activation_function,
                 dropout=0.1,
                 share_mha=False,
                 num_heads=1,
                 
                 ):
        super().__init__()
        self._share_mha = share_mha
        self._dropout = dropout
        
        if self._share_mha:
            mha = MultiheadAttention(dim,
                                     num_heads,
                                     dropout=dropout,
                                     batch_first=True)
            self._left_mha = mha
            self._right_mha = mha
        else:
            self._left_mha = MultiheadAttention(dim,
                                                num_heads,
                                                dropout=dropout,
                                                batch_first=True)
            self._right_mha = MultiheadAttention(dim,
                                                 num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)

        self._wlc = Linear(dim, dim)
        self._wrc = Linear(dim, dim)
        self._activate_func = ACT2FN[activation_function]

    def forward(self,
                hidden_rep,
                context_boundary: tuple,
                return_attentions=False):
        """
        Select the context focus and perform the context enforcement

        :param hidden_rep:
        :param context_boundary:
        :param return_attentions:
        :return:
        """

        [left_context, focus_context, right_context] = split_contexts_with_boundary(hidden_rep,
                                                                                    context_boundary, )

        # Compute the context focus attention
        left_focus, left_focus_attention = self._left_mha(focus_context,
                                                          left_context,
                                                          left_context, )
        right_focus, right_focus_attention = self._right_mha(focus_context,
                                                             right_context,
                                                             right_context, )
        
        rl_reps = self._activate_func(self._wrc(right_focus) + self._wlc(left_focus))
        focus_combined_rep = rl_reps + focus_context

        # Stitch the full reps together
        full_rep = torch.concat([left_context, focus_combined_rep, right_context], dim=1)
        if not return_attentions:
            return [focus_combined_rep,
                    full_rep], None
        else:
            return [focus_combined_rep, full_rep], [left_focus_attention, right_focus_attention]
