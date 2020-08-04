from xehm.utils.plugin import ReturnState
from xehm.utils import print_kwargs
from scipy.spatial.distance import cosine


# Compute a cosine distance as a diagnostic over two sets
def cosine_diagnostic(set_a, set_b, **kwargs) -> (int, float):
    if "debug_print" in kwargs and kwargs["debug_print"]:
        print_kwargs(**kwargs)
    score = cosine(set_a, set_b)
    return ReturnState.ok, score
