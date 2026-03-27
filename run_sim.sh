
#!/bin/bash

if [ $# -eq 0 ]; then
    echo ">> No arguments provided!"
    # echo ">> Usage: $0 - [test | ctrl (optional)] [debug (optional)] [-t <atype> - options: full, fixed, control, noStorm; test: nstest, fixedtest, debug, norm] [other arguments passed to datsim.py]"
    echo ">> Usage:  [$0:-t <atype>] [test | ctrl (optional)] [debug (optional)] [out:<outfile name>] [other arguments passed to datsim.py]"
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
    echo -e -n "\n[] [] [] args = " # 'n' tells it not to add newline at end
fi

date=$(date +'%d%b')
now=$(date +'%H:%M:%S')
dstr=$(date +'%Y%m%d')
argList=()
file="/home/wodehouse/Projects/sim_model/datsim.py"
datestr="${date:0:2}${-}${date:2}"
testOn="false"
pref=""

## (temporarily) append debugging statements after certain lines, then pipe to new shell
## mostly useful where print statement needs to be on specific line. otherwise the print functions are fine
addDebug() {
  local patt="$1" # append to lines following lines with this pattern
  # local 
  sed -n '/${patt}/p' debug_print.py
}

# datDir=$(grep )

for val in "$@"; do # loop through all CLI arguments
  #+> use wildcards to match substring:
  if [[ "$val" == *"out:"* ]]; then #+> need the double brackets
    pref+="${val:4}_" # all characters starting at index 4 (everything after 'out:'
    echo -n "$val  "
    # echo $pref
  fi
  if [ $val == "test" ]; then
    testOn="true"
    pref+="test_"
    echo -n "$val  "
    # echo $pref
  elif [ $val == "ctrl" ]; then
    pref+="ctrl_"
    echo -n "$val  "
    # echo $pref
  else ## don't add test/out/ctrl to the args to pass
    argList+=("$val")
    # pref=""
  fi
done
# echo $pref


##+> if args are being passed, will either be the test config or the full config:

echo "| args to pass to $file = ${argList[@]}"

outFile="/home/wodehouse/Projects/sim_model/${pref}${date}.out"

#
# outFile="/home/wodehouse/Projects/sim_model/logs/${pref}${date}.out"
# outFile="/home/wodehouse/Projects/sim_model/logs/test/${pref}${date}.out"
# poetry run python3 datsim.py > "$outFile" 2>&1 "${argList[@]}" &

if [ $testOn == "true" ]; then
  # echo -n " .:DEBUGGING:. "
  # addDebug '^\W*#'
  # # for now these are the same, but could add line-specific debugging?
  # sed -n '/^test:/,+16p' config.yaml | grep -v 'test' | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  sed -n '/^test:/,/debugSummary/p' config.yaml | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  # sed -n '/^test:/,/likeDir/p' config.yaml | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"

  poetry run python3 "$file" >> "$outFile" 2>&1 "${argList[@]}" &
  mypid="$!"
  echo -e "\n[] [] [] [] [] [] [] PID: $mypid [] [] [] [] [] [] output >> $outFile [] [] [] [] [] [] [] [] [] [] [] [] [] \n" 
else
  # sed -n '/^full:/,+16p' config.yaml | grep -v 'full' | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  # sed -n '/^full:/,/test/p' config.yaml | grep -v 'full|test' | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  # sed -n '/^full:/,/test/p' config.yaml | grep -vE '(full|test)' | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  sed -n '/^full:/,/debugSummary/p' config.yaml | tr '\n' ' ; ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  poetry run python3 "$file" >> "$outFile" 2>&1 "${argList[@]}" &
  mypid="$!"
  # echo -e "\n[] [] [] [] [] [] [] PID: $mypid [] [] [] [] [] [] output > $outFile (not appended) [] [] [] [] [] [] [] [] [] [] [] [] [] \n" 
  echo -e "\n[] [] [] [] [] [] [] PID: $mypid [] [] [] [] [] [] output >> $outFile [] [] [] [] [] [] [] [] [] [] [] [] [] \n" 
  echo -e "\t!!! remember-script will halt if likelihood file exists. should be saved at: /out/${dstr}/ml_val_<seed>${argList[1]}.csv\n"
  echo -e ">> ALSO remember to comment the debugging, esp. inside the optimized functions"
fi
#

# wwrite info to file:
# echo "$mypid"
# poetry run python3 datsim.py -t "control" >> 16mar.out 2>&1
# echo -e "\n\n\n[] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] []\n" >> "$outFile"  
# echo -e "\tDATE: $datestr \t\tTIME: $now \t\tPID: $mypid\tFILE: $file\n" >> "$outFile" #$! expands to PID of most recently executed background command
# echo -e "[] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] []\n" >> "$outFile"  

