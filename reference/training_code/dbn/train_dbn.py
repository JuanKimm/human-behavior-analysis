# -*- coding: utf-8 -*-
import argparse
from .pipeline import train_dbn_stage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("repeat_dir")
    parser.add_argument("split_json")
    args = parser.parse_args()
    train_dbn_stage(args.project_root, args.repeat_dir, args.split_json)

if __name__ == "__main__":
    main()
