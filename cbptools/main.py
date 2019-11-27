from .validation import Validator
from .project import Setup, display
from . import __version__, __readthedocs__
import shutil
import pkg_resources
import argparse
import logging
import errno
import os
import time
import sys
import socket


def main():
    parser = argparse.ArgumentParser(
        description='%(prog) - a python package for regional '
                    'connectivity-based parcellation',
        epilog='For a usage tutorial visit %s' % __readthedocs__
    )
    parser.add_argument('-v', '--version', action='version',
                        version='%s' % __version__)

    subcommands = parser.add_subparsers(
        title="cbptools commands", description="Entry points for cbptools"
    )
    create_command = subcommands.add_parser(
        'create', help='Create a new cbptools project'
    )
    create_command.set_defaults(run=create)
    create_command.add_argument(
        '-c', '--configfile', required=True, type=str, dest='configfile',
        help='configuration file needed to create a CBP project'
    )
    create_command.add_argument(
        '-w', '--workdir', required=True, type=str, dest='workdir',
        help='working directory where the new CBP project will be located'
    )
    create_command.add_argument(
        '-f', '--force', action='store_true', default=False, dest='force',
        help='Overwrite the current working directory if it exists'
    )
    create_command.add_argument(
        '-v', '--verbose', action='store_true', default=False, dest='verbose',
        help='Print logging to the terminal during project setup'
    )

    example_command = subcommands.add_parser(
        'example',
        help='Generate an example configuration file for the requested '
             'modality '
    )
    example_command.set_defaults(run=example)
    example_command.add_argument(
        '-g', '--get', required=True, type=str, dest='modality',
        choices=['connectivity', 'rsfmri', 'dmri'],
        help='The modality of the input data to get the configuration example '
             'file for'
    )

    args = parser.parse_args()
    args.run(args, parsers=[create_command, example_command])

    # try:
    #     args.run(args, parsers=[create_command, example_command])
    #
    # except AttributeError as exc:
    #     parser.error('too few arguments')


def create(params, parsers) -> None:
    document = params.configfile
    workdir = params.workdir
    force = params.force
    verbose = params.verbose
    parser = parsers[0]
    schema = pkg_resources.resource_filename(__name__, 'schema.yaml')

    if not os.path.isfile(document):
        parser.error(FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), document))
        return

    if os.path.exists(workdir) and len(os.listdir(workdir)) > 0 and not force:
        parser.error('Directory \'%s\' is not empty. Use -f, --force to '
                     'force overwrite an existing work_dir.' % workdir)
        return

    # Logging
    logfile = os.path.basename(document)
    logfile = os.path.splitext(logfile)[0]
    logfile = '%s_%s.log' % (logfile, str(int(time.time())))
    logging.basicConfig(
        filename=logfile,
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%I:%M:%S%p',
        level=logging.INFO
    )

    current_time = time.strftime('%b %d %Y %H:%M:%S')
    logging.info('CBP tools version %s' % __version__)
    logging.info('Setup initiated on %s in environment %s'
                 % (current_time, sys.prefix))
    try:
        # Sometimes username/hostname can't be found, it's not a big problem
        logging.info('Username of creator is \'%s\' with hostname \'%s\''
                     % (os.getlogin(), socket.gethostname()))
    except:
        pass

    if verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Document validation
    validator = Validator(schema)
    display('Validating configuration YAML document...', end='\r')
    validator.validate(document)

    if validator.errors == 0:
        document = validator.document
        display('Validating configuration YAML document {green}[OK]{endc}')
    else:
        display('Validating configuration YAML document {red}[FAILED]{endc}')
        display('{red}Project Creation Failed: Resolve all errors before '
                'continuing{endc}')
        display('{blue}Log file:{endc} %s' % logfile)
        sys.exit()

    # Run setup
    display('Validating data set files...', end='\r')
    setup = Setup(document)
    setup_valid = setup.process()

    if setup_valid:
        display('Validating data set files {green}[OK]{endc}')
        display('Creating CBPtools project directory...', end='\r')
        try:
            setup.save(workdir)
            display('Creating CBPtools project directory {green}[OK]{endc}')
        except OSError as exc:
            display('Creating CBPtools project directory {red}[FAILED]{endc}')
            display('Reason: %s' % exc)
            display('{red}Project Creation Failed: Resolve all errors before '
                    'continuing{endc}')
            display('{blue}Log file:{endc} %s' % logfile)
            sys.exit()
    else:
        display('Validating data set files {red}[FAILED]{endc}\n')
        setup.overview()
        display('\n{red}Project Creation Failed: Resolve all errors before '
                'continuing{endc}')
        display('{blue}Log file:{endc} %s' % logfile)
        sys.exit()

    # End logging and copy logfile to workdir
    logging.info('Project setup completed on %s'
                 % time.strftime('%b %d %Y %H:%M:%S'))
    logging.shutdown()
    shutil.move(logfile, os.path.join(workdir, 'log'),
                copy_function=shutil.copytree)

    setup.overview()
    display('\n{blue}Project directory:{endc} %s' % workdir)
    display('{blue}Log file:{endc} %s' % os.path.join(workdir, 'log', logfile))
    display('\nManually edit %s to execute the workflow on a cluster '
            'environment' % os.path.join(workdir, 'cluster.json'))
    sys.exit()


def example(params, parsers):
    """Copy an example configuration file to the current working directory"""
    filename = 'config_%s.yaml' % params.modality
    dest = os.path.join(os.getcwd(), filename)

    # Turn off validation logging
    logging.basicConfig(level=logging.CRITICAL)

    if os.path.exists(dest):
        path, ext = os.path.splitext(dest)
        dest = '%s({i})%s' % (path, ext)
        i = 0
        while os.path.exists(dest.format(i=i)):
            i += 1

        dest = dest.format(i=i)

    schema = pkg_resources.resource_filename(__name__, 'schema.yaml')
    validator = Validator(schema)
    validator.example(params.modality, out=dest)
    display('Created %s example configuration file at {yellow}%s{endc}'
            % (params.modality, dest))


if __name__ == "__main__":
    main()
