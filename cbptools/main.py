from cbptools.cbptools import validate_config, success_exit, fail_exit, copy_example
from cbptools import __version__, __readthedocs__
import argparse
import errno
import os
import time


def main():
    parser = argparse.ArgumentParser(
        description='%(prog) - a python package for regional '
                    'connectivity-based parcellation',
        epilog='For a usage tutorial visit %s' % __readthedocs__
    )
    parser.add_argument('-v', '--version', action='version',
                        version='%s' % __version__)

    subcommands = parser.add_subparsers(
        title="cbptools commands",
        description="Entry points for cbptools"
    )
    create_command = subcommands.add_parser(
        'create',
        help='Create a new cbptools project'
    )
    create_command.set_defaults(run=create)
    create_command.add_argument(
        '-c', '--configfile',
        required=True,
        help='configuration file needed to create a CBP project',
        type=str,
        dest='configfile'
    )
    create_command.add_argument(
        '-w', '--workdir',
        required=True,
        help='working directory where the new CBP project will be located',
        type=str,
        dest='workdir'
    )
    create_command.add_argument(
        '-f', '--force',
        action='store_true',
        default=False,
        dest='force',
        help='Overwrite the current working directory if it exists'
    )

    example_command = subcommands.add_parser(
        'example',
        help='Generate an example configuration file for the requested input '
             'data type'
    )
    example_command.set_defaults(run=copy_example)
    example_command.add_argument(
        '-g', '--get',
        required=True,
        help='What type of input data to get the configuration example '
             'file for',
        choices=['connectivity', 'rsfmri', 'dmri'],
        type=str,
        dest='input_data_type'
    )

    args = parser.parse_args()
    args.run(args, parsers=[create_command, example_command])


def create(params, parsers):
    configfile = params.configfile
    work_dir = params.workdir
    force = params.force
    parser = parsers[0]

    if not os.path.isfile(configfile):
        parser.error(FileNotFoundError(errno.ENOENT,
                                       os.strerror(errno.ENOENT), configfile))

    else:
        if os.path.exists(work_dir) and \
                len(os.listdir(work_dir)) > 0 and not force:
            parser.error('Directory \'%s\' is not empty. Use -f, --force to '
                         'force overwrite an existing work_dir.' % work_dir)

        # Create working directory
        try:
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(os.path.join(work_dir, 'log'), exist_ok=True)
        except OSError as exc:
            parser.error(OSError(exc))

        utime = str(int(time.time()))
        logfile = os.path.join(work_dir, 'log', 'project_%s.log' % utime)

        # Validate configuration file
        info = validate_config(configfile=configfile, work_dir=work_dir,
                               logfile=logfile)

        if info:
            success_exit(info, work_dir=work_dir, logfile=logfile)

        else:
            fail_exit(logfile=logfile)


if __name__ == "__main__":
    main()
