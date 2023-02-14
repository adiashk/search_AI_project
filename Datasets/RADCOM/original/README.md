This folder contains:
- radcom_dataset_with_ood_labels.csv: the dataset itself
- columns.csv: an explanation about all column names

Labels:
- You can find in the columns file a number of labels:
    - Resolution and service labels
    - OOD labels

Features:
- The features that we use based on the raw data + feature extracted from this raw data.

Samples:
- The samples extracted from encrypted network traffic of a video.
- Each sample contains up to 1 second of data
- Each video (represented as an encrypted network traffic) contains number of samples
- To get all samples extracted from specific video you can use the code_name column

