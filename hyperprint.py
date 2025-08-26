import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Batch submit Imprint jobs")
    parser.add_argument("account", help="SLURM account (e.g., p12345)")
    parser.add_argument("email", help="email for SLURM notifications")
    parser.add_argument("parent_dir", help="Parent directory containing subdirectories")
    parser.add_argument(
        "--output_root", default="outputs", help="Root output directory"
    )
    parser.add_argument("--extra_args", default="", help="Extra args for Imprint")
    args = parser.parse_args()

    if not os.path.exists(args.parent_dir):
        raise FileNotFoundError(f"Parent directory {args.parent_dir} does not exist.")

    os.makedirs(args.output_root, exist_ok=True)

    for name in [os.listdir(args.parent_dir)[0]]:
        subdir = os.path.join(args.parent_dir, name)
        print(name, subdir)
        if os.path.isdir(subdir):
            output_dir = os.path.join(args.output_root, name)
            os.makedirs(output_dir, exist_ok=True)

            # Use raw f-string to preserve bash $ and { } syntax
            SLURM_TEMPLATE = rf"""#!/bin/bash
#SBATCH --account={args.account}
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=multi-ollama
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --output=output-%j.out
#SBATCH --error=error-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user={args.email}

source_helpers () {{
  random_number () {{
    shuf -i ${{1}}-${{2}} -n 1
  }}
  export -f random_number

  port_used_python() {{
    python -c "import socket; socket.socket().connect(('${{1}}',${{2}}))" >/dev/null 2>&1
  }}

  port_used_python3() {{
    python3 -c "import socket; socket.socket().connect(('${{1}}',${{2}}))" >/dev/null 2>&1
  }}

  port_used_nc(){{ nc -w 2 "$1" "$2" < /dev/null > /dev/null 2>&1; }}
  port_used_lsof(){{ lsof -i :"$2" >/dev/null 2>&1; }}

  port_used_bash(){{
    local bash_supported=$(strings /bin/bash 2>/dev/null | grep tcp)
    if [ "$bash_supported" == "/dev/tcp/*/*" ]; then
      (: < /dev/tcp/$1/$2) >/dev/null 2>&1
    else
      return 127
    fi
  }}

  port_used () {{
    local port="${{1#*:}}"
    local host=$((expr "${{1}}" : '\(.*\):' || echo "localhost") | awk 'END{{print $NF}}')
    local port_strategies=(port_used_nc port_used_lsof port_used_bash port_used_python port_used_python3)

    for strategy in ${{port_strategies[@]}};
    do
      $strategy $host $port
      status=$?
      if [[ "$status" == "0" ]] || [[ "$status" == "1" ]]; then
        return $status
      fi
    done
    return 127
  }}
  export -f port_used

  find_port () {{
    local host="${{1:-localhost}}"
    local port=$(random_number "${{2:-2000}}" "${{3:-65535}}")
    while port_used "${{host}}:${{port}}"; do
      port=$(random_number "${{2:-2000}}" "${{3:-65535}}")
    done
    echo "${{port}}"
  }}
  export -f find_port

  wait_until_port_used () {{
    local port="${{1}}"
    local time="${{2:-30}}"
    for ((i=1; i<=time*2; i++)); do
      port_used "${{port}}"
      port_status=$?
      if [ "$port_status" == "0" ]; then
        return 0
      elif [ "$port_status" == "127" ]; then
         echo "commands to find port were either not found or inaccessible."
         echo "command options are lsof, nc, bash's /dev/tcp, or python (or python3) with socket lib."
         return 127
      fi
      sleep 0.5
    done
    return 1
  }}
  export -f wait_until_port_used
}}
export -f source_helpers
source_helpers

OLLAMA_PORT=$(find_port localhost 7000 11000)
export OLLAMA_PORT
echo $OLLAMA_PORT

module load ollama/0.11.4
module load gcc/12.3.0-gcc

export OLLAMA_HOST=0.0.0.0:${{OLLAMA_PORT}}
export SINGULARITYENV_OLLAMA_HOST=0.0.0.0:${{OLLAMA_PORT}}

ollama serve &> serve_ollama_${{SLURM_JOBID}}.log &
sleep 10

uv run ./main.py -i {subdir} -o ./{output_dir} {args.extra_args} --recursive
"""

            script_path = f"submit_{name}.sh"
            with open(script_path, "w") as f:
                f.write(SLURM_TEMPLATE)

            subprocess.run(["sbatch", script_path])
            os.remove(script_path)


if __name__ == "__main__":
    main()
