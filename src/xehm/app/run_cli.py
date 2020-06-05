# xEHM command line application
#
# v0.0.1: Goal is to setup for continuous in/out HM over one output and one input
#
import configargparse
import sys


def parse_args():
    project_url = "https://github.com/fentonscode/xEHM"

    configargparse.init_argument_parser(name="main",
                                        description=f"Evolutionary History Matching - see "
                                                    f"{project_url} for more information.",
                                        prog="xehm")

    parser = configargparse.get_argument_parser("main")
    parser.add_argument('-v', '--version', action="store_true", default=None,
                        help="Print the version information about xEHM")
    parser.add_argument('inputs', type=str, help="Input description file that contains the "
                                                 "definitions for each input variable")
    parser.add_argument('observations', type=str, help="Observation file which lists the real "
                                                       "outputs of the system")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help="Random number generator seed to use for the sampling process "
                             "(default is to use a random seed)")
    parser.add_argument('-o', '--output', type=str, default="output",
                        help="Path to the directory in which to place all "
                             "output files (default 'output').")
    args = parser.parse_args()

    # Print the version string
    if args.version:
        print("xEHM Version 0.0.1")
        sys.exit(0)

    return args, parser


def main():

    #
    # History matching workflow is as follows
    #
    # Setup
    # -----
    #
    # 1) Define the input dimensions - continuous v discrete

    pass


if __name__ == '__main__':
    main()
