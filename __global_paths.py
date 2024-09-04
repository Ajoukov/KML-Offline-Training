N = 16
M = 32
workloads = ['mixgraph', 'readrandom', 'readwhilewriting']
L = len(workloads)

INP_DIR = "input_data/"
OUT_DIR = "output_data/"
TMP_DIR = "tmp_data/"

WTS_DIR = OUT_DIR + "weights/"
TST_DIR = OUT_DIR + "tests/"

FAC = 10
EXP = 2
TARGET_ROWS = 100000

EPOCHS = 1024 * 4