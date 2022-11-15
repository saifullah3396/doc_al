#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

TYPE=standard
POSITIONAL_ARGS=()

usage()
{
    echo "Usage:"
    echo "./train.sh --type=<type>"
    echo ""
    echo " --type : Command to run. "
    echo " -h | --help : Displays the help"
    echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      shift # past argument
      usage
      exit
      ;;
    -t|--type)
      TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters



if [[ $TYPE = @(standard) ]]; then
    if [ "$TYPE" = "standard" ]; then
        LOG_LEVEL=INFO python3 $SCRIPT_DIR/../src/al/evaluate.py $@
    fi
else
  usage
  exit 1
fi
