import xehm


def main():

    emulator_form = xehm.emulators.GaussianProcess()
    emulator_diag = xehm.diagnostics.leave_one_out()[0]

    user_diagnostic = xehm.utils.build_custom_plugin("plugin_diagnostic::diagnostic_none")
    setup_parameters = {"debug_print": True}
    diagnostic_functions = user_diagnostic(setup_parameters)

    diagnostic_parameters = {}
    for f in diagnostic_functions:
        result = f(diagnostic_parameters)


if __name__ == '__main__':
    main()