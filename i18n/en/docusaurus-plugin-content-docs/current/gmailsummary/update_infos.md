---
sidebar_position: 8
---

# Integrating All

After going through the previous explanations, we've completed the development of all features. Now, we need to integrate these features to automate the entire system for future operations.

## Complete Workflow

We've written a bash script [**update_targets_infos.sh**](https://github.com/DocsaidLab/GmailSummary/blob/main/update_targets_infos.sh) to string together all the steps.

This script includes function calls and automatic pushing to GitHub.

At this stage, we're tracking several projects, including:

- **albumentations**
- **onnxruntime**
- **pytorch-lightning**
- **BentoML**
- **docusaurus**

We've also added some logging mechanisms for better tracking in the future.

```bash
#!/bin/bash

# Define directories and environment variables
origin_dir="$HOME/workspace/GmailSummary"
targ_dir="$HOME/workspace/website"
pyenv_path="$HOME/.pyenv/versions/3.10.14/envs/main/bin/python"
log_dir="$origin_dir/logs"
current_date=$(date '+%Y-%m-%d')
project_path="$targ_dir/docs/gmailsummary/news/$current_date"

# Create necessary directories
mkdir -p "$log_dir" "$project_path"

cd $origin_dir

# Specify project names list
project_names=("albumentations" "onnxruntime" "pytorch-lightning" "BentoML" "docusaurus")

for project_name in "${project_names[@]}"; do
    log_file="$log_dir/$project_name-log-$current_date.txt"
    file_name="$project_name.md"

    # Execute Python script and handle output
    {
        echo "Starting the script for $project_name at $(date)"
        $pyenv_path main.py --project_name $project_name --time_length 1
        mv "$origin_dir/$file_name" "$project_path"
        git -C "$targ_dir" add "$project_path/$file_name"
        echo "Script finished for $project_name at $(date)"
    } >> "$log_file" 2>&1

    # Check execution status
    if [ $? -ne 0 ]; then
        echo "An error occurred for $project_name, please check the log file $log_file." >> "$log_file"
    fi
done

# Push Git changes
git -C "$targ_dir" commit -m "[C] Update project report for $current_date"
git -C "$targ_dir" push
```

## Implementation Suggestions

In this project, due to integrating APIs, the entire project is filled with credentials and keys. Hence, we have some suggestions:

First and foremost, please, **do not hardcode your credentials and keys**.

Doing so will lead to the exposure of your credentials and keys, compromising the security of your emails and data.

Store these sensitive pieces of information securely and refrain from directly exposing them in any scenario.

- **Ensure Security**: Handle your `credentials.json` and API keys with care when dealing with Gmail API and OpenAI API.

The rest are minor suggestions:

- **Consider Email Diversity**: When filtering and parsing emails, consider different types of email formats and content to make the program adaptable to various situations.
- **Regular Checkups and Maintenance**: While this is an automated solution, periodic checks of execution status and updating the program to adapt to possible API changes are still necessary.