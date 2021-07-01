import csv
import sys

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.3

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python main.py data")
    
    load_data(sys.argv[1])

def load_data(filename):
    with open(filename, encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter="\t"))

        labels, evidence = ([int(r[-1]) for r in reader], [[int(rr) for rr in r[:-1]] for r in reader])

        return train_test_split(evidence, labels, test_size=TEST_SIZE, random_state=42)

if __name__ == "__main__":
    main()