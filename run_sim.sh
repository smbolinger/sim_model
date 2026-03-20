
#!/bin/bash

if [ $# -eq 0 ]; then
    echo ">> No arguments provided!"
    # echo ">> Usage: $0 - [test | ctrl (optional)] [debug (optional)] [-t <atype> - options: full, fixed, control, noStorm; test: nstest, fixedtest, debug, norm] [other arguments passed to datsim.py]"
    echo ">> Usage: $0 - [test | ctrl (optional)] [debug (optional)] [-t <atype>] [other arguments passed to datsim.py]"
    # sed -n '/SETTINGS/,+8p' config.R | grep -v 'SETTINGS' | tr '\n' '  ' | xargs echo "*** CHECK CONFIG *** " # should output the results of the pipes AFTER "check config"
    # echo ">> atype options:"
    # EXPLANATION:
    # grep -v selects the non-matching lines (invert)
    sed -n '/ANALYSIS TYPE - GROUPS/,+2p' settings.py | grep -v 'ANALYSIS TYPE - GROUPS' | tr '\n' '  ' | xargs echo ">===> atype values:" # should output the results of the pipes AFTER "check config"
# +> ANALYSIS TYPE - GROUPS:
    exit 1
else
    #echo -e "\nargument(s) passed: $1 $2 $3"
    #echo -e -n "\nargument(s) passed to shell script: $@" # 'n' tells it not to add newline at end
    echo -e -n "\n[] [] [] " # 'n' tells it not to add newline at end
fi

date=$(date +'%d%b')
now=$(date +'%H:%M:%S')
argList=()
file="/home/wodehouse/Projects/sim_model/datsim.py"
datestr="${date:0:2}${-}${date:2}"
testOn="false"

## (temporarily) append debugging statements after certain lines, then pipe to new shell
## mostly useful where print statement needs to be on specific line. otherwise the print functions are fine
addDebug() {
  local patt="$1" # append to lines following lines with this pattern
  # local 
  sed -n '/${patt}/p' debug_print.py
}

for val in "$@"; do # all CLI arguments
  if [ $val == "test" ]; then
    testOn="true"
    pref="test_"
  elif [ $val == "ctrl" ]; then
    pref="ctrl_"
  else ## don't add test to the args to pass?
    argList+=("$val")
    pref=""
  fi
done

echo "args to pass to $file = ${argList[@]}"
outFile="/home/wodehouse/Projects/sim_model/${pref}${date}.out"
#
# outFile="/home/wodehouse/Projects/sim_model/logs/${pref}${date}.out"
# outFile="/home/wodehouse/Projects/sim_model/logs/test/${pref}${date}.out"
# poetry run python3 datsim.py > "$outFile" 2>&1 "${argList[@]}" &

if [ $testOn == "true" ]; then
  # echo -n " .:DEBUGGING:. "
  # addDebug '^\W*#'
  poetry run python3 "$file" > "$outFile" 2>&1 "${argList[@]}" &
  mypid="$!"
  echo -e "\n[] [] [] [] [] [] [] PID: $mypid [] [] [] [] [] [] output >> $outFile [] [] [] [] [] [] [] [] [] [] [] [] [] \n" 
else
  poetry run python3 "$file" > "$outFile" 2>&1 "${argList[@]}" &
  mypid="$!"
  echo -e "\n[] [] [] [] [] [] [] PID: $mypid [] [] [] [] [] [] output >> $outFile [] [] [] [] [] [] [] [] [] [] [] [] [] \n" 
fi
#

# wwrite info to file:
# echo "$mypid"
# poetry run python3 datsim.py -t "control" >> 16mar.out 2>&1
# echo -e "\n\n\n[] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] []\n" >> "$outFile"  
# echo -e "\tDATE: $datestr \t\tTIME: $now \t\tPID: $mypid\tFILE: $file\n" >> "$outFile" #$! expands to PID of most recently executed background command
# echo -e "[] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] []\n" >> "$outFile"  

