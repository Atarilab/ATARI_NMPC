import casadi as cs
import numpy as np
import typeguard
import copy
from typeguard import CollectionCheckStrategy, typechecked

try:
    from pprint import pprint
except ModuleNotFoundError:
    # Error handling
    pass
# set all items for safe type check
typeguard.config.collection_check_strategy = CollectionCheckStrategy.ALL_ITEMS
