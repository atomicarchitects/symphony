import flax.core
import sys

sys.path.append("..")

from configs import graphmlp, graphnet
from train import train_and_evaluate


config = graphmlp.get_config()
train_and_evaluate(config, "graphmlp")
