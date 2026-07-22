#!/bin/bash
# tst = 0 full = 1
# con="default"
# sed -n '/ANALYSIS TYPE - GROUPS/,+2p' rsettings.py | grep -v 'ANALYSIS TYPE - GROUPS' | tr '\n' '  ' | xargs
testVals=$(sed -n '/^tests =/,+2p' rsettings.py | tr '\n' '  ') 
fullVals=$(sed -n '/fullList =/p' rsettings.py ) 
file="/home/wodehouse/Projects/sim_model/all_dsr.R"
# echo "$testVals" echo "$fullVals"
if [ $# -eq 0 ]; then
    echo ">> No arguments provided!"
    # echo -e "\t>> Usage:  [at<type>] [all_dsr.R args (optional): r<nrun>, db<debuglevel>, par<start param id>, rng<start seed> ] [out:<outfile suffix (optional)>] ["msg: message string "] [other arguments passed to datsim.py]"
    echo -e '\t>> Usage:  [at<type>] [all_dsr.R args (optional): r<nrun>, db<debuglevel>, par<start param id>, rng<start seed> ] [out:<outfile suffix (optional)>] ["msg: message string"] [other arguments passed to script]'
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
  sed -n '/^test:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; } }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** TESTING ***"
elif grep -qw "$atype" <<< "$fullVals"; then
  con="full"
  sed -n '/^full:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** FULL ***"
elif [[ "$atype" == "$control" ]]; then
  con="control"
  sed -n '/^ctrl:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** CONTROL ***"
else
  con="default"
  sed -n '/^default:/,/debugSummary/ { /^[[:blank:]]*#/! { s/#.*//; p; }  }' config.yaml | tr '\n' '|' | xargs echo -e ">===> CONFIG (***CL args or atype can override):" # should output the results of the pipes AFTER "check config"
  echo -n -e "\t*** USE DEFAULTS ***"
fi
# echo grep -w "${1:2}" fullVals
# echo "config: $con"

date=$(date +'%d%b')
now=$(date +'%H:%M:%S')
dstr=$(date +'%Y%m%d')
datestr="${date:0:2}${-}${date:2}"
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
  elif [[ "$val" == *"msg:"* ]]; then #+> need the double brackets
    argList+=("$val")
  elif [[ "$val" == *"r"* ]]; then #+> need the double brackets
    argList+=("$val")
  elif [[ "$val" == *"at"* ]]; then 
    argList+=("$val")
  elif [[ "$val" == *"par"* ]]; then 
    argList+=("$val")
  elif [[ "$val" == *"rng"* ]]; then 
    argList+=("$val")
  elif [[ "$val" == *"db"* ]]; then 
    argList+=("$val")
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

echo -e " \n [] [] [] [] $datestr - $now [] [] [] PID = $mypid [] [] [] output file = $outFile [] [] [] [] [] []" >> "$outFile"
echo -n -e "|> Rscript "$file" "${argList[@]}" >> "$outFile" 2>&1 &"
Rscript "$file" "${argList[@]}" >> "$outFile" 2>&1 &
mypid="$!"
# echo -n "args to pass to script: ${argList[@]} |"
echo " | PID = $mypid"
