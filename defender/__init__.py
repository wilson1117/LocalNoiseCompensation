from .LocalCompensation import LocalCompensation
from .OutPost import OutPost
from .GradDefense import GradDefense
from .ANP import ANP

__all__ = ['LocalCompensation', 'OutPost', 'GradDefense', 'ANP', 'get_args']

def get_args(parser):
    # Defender Config
    parser.add_argument('--defender', type=str, default=None, help='defender name')
    parser.add_argument('--calc-noise', action='store_true', help='calculate noise')

    # LocalCompensation Config
    parser.add_argument('--delta', type=float, default=2, help='max noise scale')
    parser.add_argument('--disable-compensation', action='store_true', help='disable compensation')
    parser.add_argument('--clip', type=float, default=10, help='clip upperbound')
    parser.add_argument('--warmup', type=int, default=0, help='warmup rounds')

    # OutPost Config
    parser.add_argument('--noise-base', type=float, default=0.8, help='noise base')
    parser.add_argument('--phi', type=float, default=40, help='phi')
    parser.add_argument('--prune-base', type=float, default=80, help='prune base')
    parser.add_argument('--beta', type=float, default=0.1, help='beta')

    # GradDefense Config
    parser.add_argument('--gd-unclip', dest='gd_clip', action='store_false', help='clip noise')
    parser.add_argument('--gd-slices-num', type=int, default=10, help='slices num')
    parser.add_argument('--gd-perturb-slices-num', type=int, default=5, help='perturb slices num')
    parser.add_argument('--gd-scale', type=float, default=0.01, help='scale')

    # # ANP Config
    parser.add_argument('--anp-delta', type=float, default=1e-7)
    parser.add_argument('--anp-epsilon', type=float, default=3e-3)