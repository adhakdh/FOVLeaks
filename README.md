# NeckPass
The files in this repository are for the paper: **FOVLeaks: Exploiting Field-of-View Dynamics for Input Inference in VR/AR**

We provide the following three compressed data packages:
1. **ExperimentalResults**: This package contains the data sources for all scenarios presented in the paper. It includes processed data and trained models. For each scenario, simply `python run_1.py` to obtain the result data and `python run_2.py` to get the structured results presentation shown in the paper.
1. **SourceCode**: This package provides the source code for the technical approach mentioned in the paper. It includes data processing methods. By replacing the dataset with the corresponding scenario dataset mentioned in the paper and running `python run.py`, you can achieve the data processing results presented in the paper.
1. **Datasets**: This package includes the datasets for each scenario in the paper.

**Detailed Explanation of the First Data Package:.** *ExperimentalResults*. The directory structure is as follows:

```
requirements.txt
Result.xlsx
lib
├── anchor_data
├── savedata
├── lib.py
├── lib1.py
├── lib2.py
└── lib3.py
5.3_CharacterInference
├── Dataset_FOVLeaks
├── run_1.py
└── run_2.py
5.4_PasswordandPINInference
├── Dataset_FOVLeaks
├── run_1.py
└── run_2.py
5.5_WordInference
├── Dataset_FOVLeaks
├── run_1.py
└── run_2.py
5.6_WebsiteURLInference
├── Dataset_FOVLeaks
├── run_1.py
└── run_2.py
llm
├── Dataset_FOVLeaks.xlsx
├── run_llm.py
└── run_top3.py
```

**Step 1.** In the first-level directory, the `requirements.txt` file lists all the libraries required for this folder. To install the dependencies, execute the following command (My Python version 3.9.11):
> `pip install -r requirements.txt `

**Step 2.** The `Result.xlsx` file provides the data presented in each scenario of the paper, serving as a reference outline for readers. 

**Step 3.** The `lib` folder contains implementations of library functions. The remaining folders correspond to the scenario resource files, named to align with the sections in the paper for easy reference. 

**Step 4.** For each scenario (paper section), the folder contains 'Dataset_FOVLeaks', which is the corresponding dataset. In addition, two files: `run_1.py` and `run_2.py`. Running these two files allows you to obtain the results, consistent with those provided in `Result.xlsx` and in the paper. The corresponding commands are as follows:
> `python run_1.py`
> `python run_2.py`

**Step 5.** For 'llm' path, the folder contains 'Dataset_FOVLeaks.xlsx', which is the corresponding dataset. In addition, two files: `run_llm.py` and `run_top3.py`. Running these two files allows you to obtain the results, consistent with those provided in `Result.xlsx` and in the paper. The corresponding commands are as follows:
> `python run_llm.py`
> `python run_top3.py`

**If you encounter any difficulties, please don't hesitate to reach out for assistance. Thank you sincerely for your interest, time and patience.**