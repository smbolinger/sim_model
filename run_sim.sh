
#!/bin/bash

if [ $# -eq 0 ]; then
    echo ">> No arguments provided!"
    echo ">> Usage: $0 - [test (optional)] [debug (optional)] [-t <atype> - options: full, fixed, control, noStorm; test: nstest, fixedtest, debug, norm] [other arguments passed to datsim.py]"
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
outFile="/home/wodehouse/Projects/sim_model/${date}.out"
testOn="false"
datestr="${date:0:2}${-}${date:2}"
for val in "$@"; do # all CLI arguments
  if [ $val == "test" ]; then
    testOn="true"
  else ## don't add test to the args to pass?
    argList+=("$val")
  fi
done

echo "args = ${argList[@]}"

if [ $testOn == "true" ]; then
  outFile="/home/wodehouse/Projects/sim_model/test-${date}.out"
fi

# poetry run python3 datsim.py > "$outFile" 2>&1 "${argList[@]}" &
poetry run python3 "$file" > "$outFile" 2>&1 "${argList[@]}" &
mypid="$!"
# echo "$mypid"
# poetry run python3 datsim.py -t "control" >> 16mar.out 2>&1
echo -e "\n\n\n[] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] []\n" >> "$outFile"  
echo -e "\tDATE: $datestr \t\tTIME: $now \t\tPID: $mypid\tFILE: $file\n" >> "$outFile" #$! expands to PID of most recently executed background command
echo -e "[] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] []\n" >> "$outFile"  

echo -e "\n[] [] [] [] [] [] [] PID: $mypid [] [] [] [] [] [] output >> $outFile [] [] [] [] [] [] [] [] [] [] [] [] [] \n" 
