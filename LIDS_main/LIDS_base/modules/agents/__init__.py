from .rnn_agent import RNNAgent
from .strat_rnn_agent import RNNAgent_with_strategy
REGISTRY = {}


REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_w_strategy"]=RNNAgent_with_strategy
