#!/bin/bash
# tst = 0 full = 1
# con="default"
# sed -n '/ANALYSIS TYPE - GROUPS/,+2p' rsettings.py | grep -v 'ANALYSIS TYPE - GROUPS' | tr '\n' '  ' | xargs
testVals=$(sed -n '/^tests =/,+2p' rsettings.py | tr '\n' '  ') 
fullVals=$(sed -n '/fullList =/p' rsettings.py ) 
# file="/home/wodehouse/Projects/sim_model/all_dsr.R"
file="/home/wodehouse/Projects/sim_model/main.R"
# echo "$testVals" echo "$fullVals"
if [ $# -eq 0 ]; then
    echo ">> No arguments provided!"
    # echo -e "\t>> Usage:  [at<type>] [all_dsr.R args (optional): r<nrun>, db<debuglevel>, par<start param id>, rng<start seed> ] [out:<outfile suffix (optional)>] ["msg: message string "] [other arguments passed to datsim.py]"
    echo -e '\t>> Usage:  [at<type>] [all_dsr.R args (optional): rep<nrun>, db<debuglevel>, par<start param id>, rng<start seed>, cl<cpu limit>, ml<memory limit AS DECIMAL>] [nomc] [nolx] [pred] [savend] [out:<outfile suffix (optional)>] ["msg: message string"] [other arguments passed to script]'
    # echo ">> analysis type options:"
    sed -n '/ANALYSIS TYPE - GROUPS/,+5p' rsettings.py | grep -v 'ANALYSIS TYPE - GROUPS' | tr '\n' '  ' | xargs echo ">===> atype values: " # should output the results of the pipes AFTER "check config"
    echo -e '\t>> run again with "help" for more info on analysis types'
    exit 1
elif [ $1 == "help" ]; then
    echo -n -e ">> help-analysis types:\t\t"
    exit 1
else
    #echo -e "\nargument(s) passed: $1 $2 $3"
    #echo -e -n "\nargument(s) passed to shell script: $@" # 'n' tells it not to add newline at end
    # echo -e -n "[] [] [] run $file ; args = $@" # 'n' tells it not to add newline at end
    echo -e -n ">>" # 'n' tells it not to add newline at end
fi

# should work for multiple instances of start and end (exclude everything in between and send the rest to a new file):
# export MY_REGEX="(?<=code is )\w+"
# echo "$TEXT" | perl -ne 'print $1 if /$ENV{MY_REGEX}/'
# perl -0777 -pe 's/START_PATTERN.*?END_PATTERN//gs' input.txt > output.txt
# flist = ("observer.py" "makeNests.py" "all_dsr.R" "lexp_fun.R")

flist=("observer.py" "makeNests.py" "lexp_fun.R")

for fname in "${flist[@]}"; do
  perl -0777 -pe 's/#-\*~*.*?#-=~*//gs' "$fname" > "nodebug_$fname"
done

# echo "config: $con"
# echo "${1:2}"
atype="${1:2}"
control="control"
# echo " $atype"
if grep -qw "$atype" <<< "$testVals"; then
  con="test"
  ## if line does not contain 0+ blanks followed by # at beginning of line, print it
  ## also substitute "" for anything follpwing #
  # sed -n '/^[[:blank:]]*#/!{/^test:/,/debugSummary/p}' config.yaml | tr '\n' '  ' | xargs echo -e "\n>===> CONFIG:" # should output the results of the pipes AFTER "check config"
  # config=$(sed -n '/^test:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|')# should output the results of the pipes AFTER "check config"
  configStr=$(sed -n '/^test:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' ' ' ) # should output the results of the pipes AFTER "check config"
  sed -n '/^test:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; } }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** TESTING ***"
elif grep -qw "$atype" <<< "$fullVals"; then
  con="full"
  configStr=$(sed -n '/^full:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' ' ' ) # should output the results of the pipes AFTER "check config"
  sed -n '/^full:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** FULL ***"
elif [[ "$atype" == "$control" ]]; then
  con="control"
  configStr=$(sed -n '/^ctrl:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' ' ' ) # should output the results of the pipes AFTER "check config"
  sed -n '/^ctrl:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** CONTROL ***"
else
  con="default"
  configStr=$(sed -n '/^default:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' ' ' ) # should output the results of the pipes AFTER "check config"
  sed -n '/^default:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** USE DEFAULTS ***"
fi
# echo grep -w "${1:2}" fullVals
# echo "config: $configStr"
# echo "$config"

clim=150
date=$(date +'%d%b')
now=$(date +'%H:%M:%S')
dstr=$(date +'%Y%m%d')
# datestr="${date:0:2}${-}${date:2}"
datestr="${date:0:2}-${date:2}"
argList=() ## args that will be passed
rrList=()
# testOn="false"
pref=""

# arg_pref

addDebug() {
  local patt="$1" # append to lines following lines with this pattern
  # local 
  sed -n '/${patt}/p' debug_print.py
}

for val in "$@"; do # loop through all CLI arguments
  if [[ "$val" == *"out:"* ]]; then #+> need the double brackets
    pref+="-${val:4}" # all characters starting at index 4 (everything after 'out:'
  elif [[ "$val" == *"cl"* ]]; then #+> need the double brackets
    clim=("${val:2}")
  elif [[ "$val" == *"ml"* ]]; then #+> need the double brackets
    export XLA_PYTHON_CLIENT_MEM_FRACTION=("${val:2}")
  elif [[ "$val" == *"other:"* ]]; then #+> need the double brackets
    argList+=("--oval=\"${val:6}\"")
  elif [[ "$val" == *"msg:"* ]]; then #+> need the double brackets
    argList+=("--msg=\"${val:4}\"")
  elif [[ "$val" == *"rep"* ]]; then #+> need the double brackets
    argList+=("--nreps=${val:3}")
  elif [[ "$val" == *"at"* ]]; then 
    argList+=("--atype=\"${val:2}\"")
  elif [[ "$val" == *"par"* ]]; then 
    argList+=("--par=${val:3}")
  elif [[ "$val" == *"rng"* ]]; then 
    argList+=("--rng=${val:3}")
  elif [[ "$val" == *"db"* ]]; then 
    argList+=("--db=${val:2}")
  elif [[ "$val" == *"savend"* ]]; then 
    argList+=("--savend")
  elif [[ "$val" == *"nomc"* ]]; then 
    argList+=("--nomc")
  elif [[ "$val" == *"nolx"* ]]; then 
    argList+=("--nolx")
  elif [[ "$val" == *"pred"* ]]; then 
    argList+=("--pred")
  else
    pref+="-$val" # all characters starting at index 4 (everything after 'out:'
    # rrList+=("$val")
  fi
done

# echo "args to pass to rr: ${rrList[@]} |"
# suff="-${rrList[0]}" 
# outFile="/home/wodehouse/Projects/sim_model/${date}-r${suff}.out"
outFile="/home/wodehouse/Projects/sim_model/${date}-r${pref}.out"
# echo "outfile = $outFile"
# echo "${argList[@]}"

# echo -n -e "|> Rscript "$file" "${argList[@]}" >> "$outFile" 2>&1 &"
echo -e "|> Rscript "$file" "${argList[@]}" >> "$outFile" 2>&1 &"
Rscript "$file" "${argList[@]}" >> "$outFile" 2>&1 &
# mypid="$!"
mypid=$!
echo -e "\n\n+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + " >> "$outFile"
echo -e -n "\n[] [] [] [] $datestr - $now [] [] [] PID = $mypid [] [] [] output file = $outFile [] [] [] [] [] []" >> "$outFile"
# echo -n "args to pass to script: ${argList[@]} |"
# echo -n " | PID = $mypid"
echo -e -n "\t\t>>>> PID = $mypid"
# echo -e -n ">> $datestr - $now >>>> PID = $mypid" >> "PIDs.txt"
sleep 2
cpulimit -p "$mypid" -l "$clim" &
clpid=$!

echo " ** cpulimit PID = $clpid [] [] []" >> "$outFile"
echo -n " | cpulimit PID = $clpid"
echo -n " | limiting CPU to $clim % | "

# rngSeed=$(sed -n '/^rngSeed: /,+2p' rsettings.py | tr '\n' '  ') 
# rngSeed=$(grep -oP "rngSeed:\s+\K\w+" "$config") 
# rngSeed=$(grep -oP '^rngSeed:\s+\K\d+' <<< "$config") 
# sfate=$(grep -oP '^stormFate:\s+\K\d' <<< "$config") 
# mcType=$(grep -oP '^mcType:\s+\K\d' <<< "$config") 
# numNests=$(grep -oP '^numNests:\s+\K\d' <<< "$config") 
# echo "seed= $rngSeed ; sfate= $sfate ; mctype= $mcType ; num nests= $numNests"
echo -e -n ">> $datestr - $now >>>> PID: $mypid | cpulimit PID: $clpid " >> "PIDs.txt"
echo -e "|> Rscript "$file" "${argList[@]}" >> "$outFile" 2>&1 &" >> "PIDs.txt"
# echo -e "$configStr \n" >> "PIDs.txt"
echo -e "  >> config (before changes): $configStr \n" >> "PIDs.txt"
