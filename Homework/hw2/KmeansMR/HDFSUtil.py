import os
import subprocess


def read_lines(hdfs_path, hadoop_prefix=os.environ.get("HADOOP_PREFIX")):
    hadoop = os.path.join(hadoop_prefix, "bin", "hadoop")
    cat = subprocess.Popen([hadoop, "fs", "-cat", hdfs_path], stdout=subprocess.PIPE)
    for line in cat.stdout:
        yield line
