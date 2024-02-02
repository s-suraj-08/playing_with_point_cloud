from pathlib import Path
import subprocess

path = Path("./data")
path_ls = list( path.glob('*.ply'))
path_ls.sort()


savepath = "./out"
for p in path_ls:
    COMMAND = ["python", "main.py", "-p", f"{str(p)}", "--random_transform", "--savepath", f"{savepath}/{p.name}"]
    subprocess.run(COMMAND)