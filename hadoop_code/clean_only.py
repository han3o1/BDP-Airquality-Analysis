# -*- coding: utf-8 -*-

import sys

def clean_field(v):
    v = v.strip()

    if v == "" or v == "-999":
        return ""

    try:
        f = float(v)
        if f < 0:
            return ""
        return v
    except:
        return ""

def process_line(line):
    parts = line.strip().split(",")

    if len(parts) != 11:
        return None

    parts[4] = clean_field(parts[4])
    parts[5] = clean_field(parts[5])
    parts[6] = clean_field(parts[6])
    parts[7] = clean_field(parts[7])
    parts[8] = clean_field(parts[8])
    parts[9] = clean_field(parts[9])

    return ",".join(parts)

def main():
    if len(sys.argv) != 3:
        print "Usage: python clean_only.py input.csv output.csv"
        sys.exit(1)

    fin = open(sys.argv[1], "r")
    fout = open(sys.argv[2], "w")

    header = fin.readline()
    fout.write(header)

    total = 0
    valid = 0

    for line in fin:
        total += 1
        cleaned = process_line(line)
        if cleaned:
            fout.write(cleaned + "\n")
            valid += 1

    fin.close()
    fout.close()

    print "Total:", total
    print "Valid:", valid
    print "Dropped:", total - valid

if __name__ == "__main__":
    main()
