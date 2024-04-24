---
sidebar_position: 10
---

# Conclusion

With this project, we've once again successfully increased efficiency (or rather, indulged in some laziness).

We hope this solution can help those with similar needs, and we look forward to exploring more automation solutions to optimize daily workflows in the future.

Our daily updates are directly pushed on this webpage. If you're interested, you can directly check the **Daily News** section.

## FAQ

1. **Why not use GPT-4?**

    Because it's expensive. Although the generated content is great, the price is **20 times** that of GPT-3.5.

    While we like to be lazy, we can't incur too much expense.

2. **Why isn't there an English version of the daily news?**

    Because running OpenAI API incurs expenses for each token, so we're currently only running Chinese. We'll consider English... emm, later.

3. **Isn't your email content confidential?**

    No, these emails are all public. You just need to go to the GitHub page of those open-source projects to see all the content, although we guess you don't have the patience to read through it all.

4. **How do you specify which emails to analyze?**

    In `update_targets_infos.py`, you can modify `project_names` to set the projects you want to analyze.

    ```bash
    # update_targets_infos.py
    # Specify the list of project names
    project_names=("albumentations" "onnxruntime") # <-- Modify this line
    ```
