import argparse
import csv
import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results, "*.csv")))
    if not paths:
        print("no csv files in", args.results)
    for path in paths:
        with open(path) as f:
            rows = list(csv.reader(f))
        print("### " + os.path.basename(path))
        print()
        print("| " + " | ".join(rows[0]) + " |")
        print("|" + "---|" * len(rows[0]))
        for row in rows[1:]:
            print("| " + " | ".join(row) + " |")
        print()


if __name__ == "__main__":
    main()
