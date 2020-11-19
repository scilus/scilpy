#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to search a specific database hosted on Beluga or Braindata.
"""

import argparse
import re

import pandas as pd


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--name", help="Name of the database")
    p.add_argument("--shells", nargs="+", help="Shells wanted in the datases")
    p.add_argument(
        "--nb_sub",
        default=0,
        type=int,
        help="Minimum number of subjects wanted.",
    )
    p.add_argument(
        "--nb_ses",
        default=0,
        type=int,
        help="Minimum number of sessions wanted.",
    )
    p.add_argument(
        "--nb_run", default=0, type=int, help="Minimum number of runs wanted."
    )
    p.add_argument(
        "--is_healthy",
        action="store_true",
        help="Database must contain healthy subjects.",
    )
    p.add_argument(
        "--is_challenge",
        action="store_true",
        help="Database must come from a challenge.",
    )

    p.add_argument(
        "--csv",
        default="/braindata/utils/databases.csv",
        help="Path of the CSV containing all the databases " "informations.",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.name:
        name_regexes = re.compile(
            "(" + re.escape(args.name) + ")", re.IGNORECASE
        )
        for i, name in enumerate(df["Name"]):
            if not len(name_regexes.findall(name)):
                df = df.drop(df.loc[df["Name"] == name].index)
        df = df.reset_index(drop=True)

    if args.shells:
        to_keep = []
        for wanted_shell in args.shells:
            for i, shells in enumerate(df["Shells"]):
                for shell in shells.split(","):
                    if shell == wanted_shell:
                        to_keep.append(df.loc[df["Shells"] == shells].index[0])
        to_remove = list(set(range(len(df))).difference(to_keep))
        df = df.drop(to_remove)
        df = df.reset_index(drop=True)

    df = df.drop(df.loc[df["Nb Subjects"].astype(int) < args.nb_sub].index)
    df = df.reset_index(drop=True)
    df = df.drop(df.loc[df["Nb Sessions"].astype(int) < args.nb_ses].index)
    df = df.reset_index(drop=True)
    df = df.drop(df.loc[df["Nb Runs"].astype(int) < args.nb_run].index)
    df = df.reset_index(drop=True)

    if args.is_healthy:
        df = df.drop(
            df.loc[df["Is Healthy"].astype(bool) != args.is_healthy].index
        )
        df = df.reset_index(drop=True)

    if args.is_challenge:
        df = df.drop(
            df.loc[df["Is Challenge"].astype(bool) != args.is_challenge].index
        )
        df = df.reset_index(drop=True)

    if len(df):
        print(df.to_string())
    else:
        print("No databases match your request.")


if __name__ == "__main__":
    main()
