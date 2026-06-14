#!/usr/bin/env python3
"""Quest3 UDP -> Dobot ServoP tool-frame teleoperation.

This entry point intentionally reuses ``servoj_toolframe_teleop`` and forces
the servo mode to ``cartesian`` so the final command path is ServoP.
"""

import sys

from servoj_toolframe_teleop import main


def _force_servop_mode(argv):
    filtered = [argv[0]]
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--servo-mode":
            skip_next = True
            continue
        if arg.startswith("--servo-mode="):
            continue
        filtered.append(arg)

    if "--help" not in filtered and "-h" not in filtered:
        filtered.extend(["--servo-mode", "cartesian"])
    return filtered


if __name__ == "__main__":
    sys.argv = _force_servop_mode(sys.argv)
    main()
