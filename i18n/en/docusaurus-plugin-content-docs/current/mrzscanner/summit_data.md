---
sidebar_position: 9
---

# Submission

In the real world, there's a wide variety of cases, and you might encounter situations where things don’t work as expected.

The same applies to our model, as it may not handle every scenario perfectly.

If you find that our model struggles to handle certain cases, we encourage you to provide datasets. We will use the datasets you submit to fine-tune and optimize our model.

We sincerely appreciate your willingness to share datasets, and we will prioritize testing and integration as soon as we receive them.

## Format Guidelines

For this task, you need to submit:

- **MRZ images with text labels, similar to the MIDV-2020 dataset we mentioned earlier.**

---

For example, if you have an image with an MRZ section, the corresponding label data might look like this:

```json
{
  "img_01": {
    "img_path": "path/to/image.jpg",
    "mrz_1": "P<USALAST<<FIRST<MIDDLE<NAME<<<<<<<<<<<<<<<<",
    "mrz_2": "1234567890USA1234567890<<<<<<<<<<<<<<<<<<<<4"
  },
  "img_02": {
    "img_path": "path/to/image.jpg",
    "mrz_1": "P<USALAST<<FIRST<MIDDLE<NAME<<<<<<<<<<<<<<<<",
    "mrz_2": "1234567890USA1234567890<<<<<<<<<<<<<<<<<<<<4"
  }
}
```

---

We recommend uploading the dataset to your Google Drive and sharing the link with us via [**email**](#contact-us). Upon receiving your dataset, we will conduct testing and integration as soon as possible. If the dataset does not meet our requirements, we will notify you promptly.

## Frequently Asked Questions

1. **Will submitting my dataset improve the model performance?**

   - It's uncertain. Although the model will see the data you provide, it doesn’t guarantee that the dataset’s characteristics will significantly influence the model. Having more data is better than none, but it may not lead to a dramatic improvement.

2. **How important are file names?**

   - File names are not the primary focus, as long as they correctly link to the corresponding images.

3. **What image format do you recommend?**
   - We recommend using the jpg format to save space.

---

## Contact Us

If you need further assistance, feel free to reach out to us via email: **docsaidlab@gmail.com**
