import os
import argparse
import subprocess

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --account={account}  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need - for AlphaFold this should always be one
#SBATCH --ntasks-per-node=4 ## how many cpus or processors do you need on each computer
#SBATCH --job-name=multi-ollama ## When you run squeue -u <NETID> this is how you can identify the job
#SBATCH --time=0:30:00 ## how long does this need to run
#SBATCH --mem=16GB ## how much RAM do you need per node (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --gres=gpu:1 ## type of GPU requested, and number of GPU cards to run on
#SBATCH --output=output-%j.out ## standard out goes to this file
#SBATCH --error=error-%j.err ## standard error goes to this file
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user={email} ## your email, non-Northwestern email addresses may not be supported


# Source in all the helper functions - No need to change any of this
source_helpers () {
  # Generate random integer in range [$1..$2]
  random_number () {
    shuf -i ${1}-${2} -n 1
  }
  export -f random_number

  port_used_python() {
    python -c "import socket; socket.socket().connect(('$1',$2))" >/dev/null 2>&1
  }

  port_used_python3() {
    python3 -c "import socket; socket.socket().connect(('$1',$2))" >/dev/null 2>&1
  }

  port_used_nc(){
    nc -w 2 "$1" "$2" < /dev/null > /dev/null 2>&1
  }

  port_used_lsof(){
    lsof -i :"$2" >/dev/null 2>&1
  }

  port_used_bash(){
    local bash_supported=$(strings /bin/bash 2>/dev/null | grep tcp)
    if [ "$bash_supported" == "/dev/tcp/*/*" ]; then
      (: < /dev/tcp/$1/$2) >/dev/null 2>&1
    else
      return 127
    fi
  }

  # Check if port $1 is in use
  port_used () {
    local port="${1#*:}"
    local host=$((expr "${1}" : '\(.*\):' || echo "localhost") | awk 'END{print $NF}')
    local port_strategies=(port_used_nc port_used_lsof port_used_bash port_used_python port_used_python3)

    for strategy in ${port_strategies[@]};
    do
      $strategy $host $port
      status=$?
      if [[ "$status" == "0" ]] || [[ "$status" == "1" ]]; then
        return $status
      fi
    done

    return 127
  }
  export -f port_used

  # Find available port in range [$2..$3] for host $1
  # Default: [2000..65535]
  find_port () {
    local host="${1:-localhost}"
    local port=$(random_number "${2:-2000}" "${3:-65535}")
    while port_used "${host}:${port}"; do
      port=$(random_number "${2:-2000}" "${3:-65535}")
    done
    echo "${port}"
  }
  export -f find_port

  # Wait $2 seconds until port $1 is in use
  # Default: wait 30 seconds
  wait_until_port_used () {
    local port="${1}"
    local time="${2:-30}"
    for ((i=1; i<=time*2; i++)); do
      port_used "${port}"
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
  }
  export -f wait_until_port_used

}
export -f source_helpers

source_helpers

# Find available port to run server on
OLLAMA_PORT=$(find_port localhost 7000 11000)
export OLLAMA_PORT
echo $OLLAMA_PORT


module load ollama/0.11.4
module load gcc/12.3.0-gcc

export OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} #what should our IP address be?
export SINGULARITYENV_OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}

#start Ollama service
ollama serve &> serve_ollama_${SLURM_JOBID}.log &
sleep 10
#wait until Ollama service has been started


#Run the python script
uv run ./main.py -i ./{name} -o ./{output_dir} {extra_args}
"""


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

    for name in os.listdir(args.parent_dir):
        subdir = os.path.join(args.parent_dir, name)
        if os.path.isdir(subdir):
            output_dir = os.path.join(args.output_root, name)
            script_content = SLURM_TEMPLATE.format(
                account=args.account,
                email=args.email,
                name=name,
                input_dir=subdir,
                output_dir=output_dir,
                extra_args=args.extra_args,
            )
            script_path = f"submit_{name}.sh"
            with open(script_path, "w") as f:
                f.write(script_content)
            subprocess.run(["sbatch", script_path])
            os.remove(script_path)


if __name__ == "__main__":
    main()

