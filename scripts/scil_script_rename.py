#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from scilpy import SCILPY_ROOT
from scilpy.io.utils import add_verbose_arg, add_overwrite_arg
from shutil import move


LEGACY_SCRIPT_RENAMING_TEMPLATE = """\
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.{new_name} import main as new_main


DEPRECATION_MSG = '''
This script has been renamed {new_name}.py. Please change
your existing pipelines accordingly.
'''

@deprecate_script("{existing_name}.py", DEPRECATION_MSG, '{version}')
def main():
    new_main()


if __name__ == "__main__":
    main()

"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('existing_name', help='Name of an existing scilpy '
                                           'script to rename.')
    p.add_argument('new_name', help='Name of the new script.')
    p.add_argument('deprecation_version', help='Version at which the script '
                                               'will display deprecation.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    script_home = os.path.join(SCILPY_ROOT, 'scripts')
    script_path = os.path.join(script_home, f"{args.existing_name}.py")
    new_script_path = os.path.join(script_home, f"{args.new_name}.py")
    legacy_script_path = os.path.join(script_home, 'legacy',
                                      f"{args.existing_name}.py")

    if not os.path.exists(script_path):
        raise ValueError(f"Script {args.existing_name} does not exist.")

    if os.path.exists(new_script_path):
        raise ValueError(f"Script {args.new_name} already exists.")

    with open(legacy_script_path, 'w') as new_script:
        new_script.write(LEGACY_SCRIPT_RENAMING_TEMPLATE.format(
            existing_name=args.existing_name, new_name=args.new_name,
            version=args.deprecation_version))

    os.chmod(legacy_script_path, 0o755)
    move(script_path, new_script_path)


if __name__ == "__main__":
    main()
