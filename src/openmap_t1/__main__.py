import argparse
import pathlib
import sys

from transformers import HfArgumentParser

from openmap_t1.commands.parcellation import ParcellationArgs, run_parcellations


def run():
    parser = argparse.ArgumentParser(
        description="Use this to run inference with OpenMAP-T1."
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.add_parser(name="parcellation", help="Run parcellation process.")

    supported_subcommands = list(subparsers.choices.keys())

    for argument in sys.argv[1:]:
        if argument in supported_subcommands:
            subcommand_args = parser.parse_args([argument])
            sys.argv.remove(argument)
            break

    if not hasattr(subcommand_args, "subcommand"):
        parser.print_help()
        sys.exit(1)

    if subcommand_args.subcommand == "parcellation":
        subparser = HfArgumentParser(dataclass_types=ParcellationArgs)
        (args,) = subparser.parse_args_into_dataclasses()
        return run_parcellations(args)
    else:
        raise ValueError(f"Unsupported subcommand: {subcommand_args.subcommand}")
