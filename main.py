from __future__ import print_function
import _init_paths
from attack_model import *
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import flags
FLAGS = flags.FLAGS

def main(argv = None):
    flags.print_attack_flags()
    AttackGraph()

    attack = Attack()
    attack.optimize()

if __name__ == '__main__':
    app.run()