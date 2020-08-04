import xehm
import numpy as np


def main():

    user_diagnostic = xehm.utils.build_custom_plugin("plugin_diagnostic::cosine_diagnostic")
    extra_parameters = {"debug_print": True}
    a = np.random.uniform(0.0, 1.0, size=5)
    b = np.random.uniform(0.0, 1.0, size=5)
    result, score = user_diagnostic(set_a=a, set_b=b, **extra_parameters)
    if result != xehm.utils.ReturnState.ok:
        print("The diagnostic failed")
    else:
        print(f"Dignostic score: {score}")


if __name__ == '__main__':
    main()