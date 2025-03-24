#!/bin/bash

#Script to configure artifacts

#Copyright 2025 DFX5.

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

echo "Starting automated artifacts configuration"

#helpFunction()
#{
#   echo ""
#   echo "Usage: $0 -f CHANGED_FILES_ARRAY -a ARTIFACT_FILE_NAME"
#   echo -e "\t-f Array of changed files"
#   echo -e "\t-a Artifact file name"
#   exit 1 
#}
#
#while getopts "f:a:" opt
#do
#   case "$opt" in
#      f ) CHANGED_FILES_ARRAY=($OPTARG) ;;
#      a ) ARTIFACT_FILE_NAME="$OPTARG" ;;
#      ? ) helpFunction ;; 
#   esac
#done
#
#if [ -z "$CHANGED_FILES_ARRAY" ] || [ -z "$ARTIFACT_FILE_NAME" ]
#then
#   echo "Empty changed files array or artifact file name";
#   helpFunction
#fi


# Read array of changed files 
CHANGED_FILES_ARRAY=($1)
# Read outputfile name
ARTIFACT_FILE_NAME="$2"

deployment_size=${#CHANGED_FILES_ARRAY[@]}

# Check number of chaged files in the current commit
if [ $deployment_size -eq 0 ]
then
  echo "The zero input files for this $ARTIFACT_FILE_NAME artifact , nothing to do, skipping."
  exit 0
else
  echo "The number of input files is: $deployment_size, procceding with artifact filtering/generation"
  # Iterate over changed files, then, adding ony JSON files to artifact
  for file in "${CHANGED_FILES_ARRAY[@]}"
  do
    if [[ $file =~ \.json$  ]];
    then
     echo "   JSON file:  $file"
     echo "$file" >> $ARTIFACT_FILE_NAME
    fi
  done 
  # Check if artifact was created, aborting if not JSON files provided
  if [ -e "$ARTIFACT_FILE_NAME" ]; then
    printf "Artifact $ARTIFACT_FILE_NAME has been created\n"
  else
    printf "No JSON files detected, $ARTIFACT_FILE_NAME skipped\n"
    exit 0
  fi
fi

