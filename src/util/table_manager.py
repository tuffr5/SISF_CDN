import sys
import sqlite3
import argparse

FNAME = "traces.sql"
debug = True

##################################################
# Parse Commandline Arguments
##################################################

parser = argparse.ArgumentParser(
    prog="Table Manager",
    description="A tool for manipulating SISF_CDN SWC tables",
    epilog="",
)

parser.add_argument(
    "-i",
    "--init",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Creates a new table.",
)

parser.add_argument(
    "-l",
    "--list",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Lists neurons found in the table.",
)

parser.add_argument(
    "-g", "--get_neuron", type=int, help="Download a neuron ID as a SWC file."
)

parser.add_argument(
    "-f",
    "--override_file_name",
    type=str,
    default="traces.sql",
    help="Manually override the sqlite database name.",
)

args = parser.parse_args()

FNAME = args.override_file_name

if debug:
    print("Args:", args)

##################################################
# Init Table
##################################################
if args.init:
    print(f"Creating a new table [{FNAME}]:")

    with sqlite3.connect(FNAME) as con:
        cur = con.cursor()

        print("\tAdding SWC table...")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS SWC(
                    I INT                  NOT NULL,
                    NEURONID INT           NOT NULL,
                    PARENTID INT           NOT NULL,
                    X REAL              NOT NULL,
                    Y REAL              NOT NULL,
                    Z REAL              NOT NULL,
                    R REAL                 NOT NULL,
                    T INT                  NOT NULL,
                    USERID INT             NOT NULL,
                    TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
            """
        )

        print("\tAdding Neuron index table...")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS NEURONS(
						   SOMAX REAL,
						   SOMAY REAL,
						   SOMAZ REAL,
						   CELLTYPE INT,
						   NOTES TEXT,
						   TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP
						   );
            """
        )
        con.commit()

    print("\tComplete.")

    sys.exit(0)

if args.list:
    print(f"Neurons in [{FNAME}]:")
    print("=" * 40)

    with sqlite3.connect(FNAME) as con:
        cur = con.cursor()
        cur.execute("SELECT rowid,* FROM NEURONS;")
        for i, _somax, _somay, _somaz, _celltype, _notes, date in cur.fetchall():
            print(f"Neuron {i} made on {date}")

    sys.exit(0)

if args.get_neuron is not None:
    # print("Getting")

    with sqlite3.connect(FNAME) as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT I,T,X,Y,Z,R,PARENTID FROM SWC WHERE NEURONID={args.get_neuron};"
        )

        for res in cur.fetchall():
            print(",".join(map(str, res)))

    sys.exit(0)
