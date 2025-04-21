import json
import pandas as pd
from pathlib import Path
import re
import numpy as np
import warnings
import torch
from datasets import Dataset
#from MRrep_cleaning_helper import *
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from mayo_prostate import nlp
from sklearn.metrics import confusion_matrix
from itertools import product
import json
from MRrep_fixing_treatment_status_byKurata import is_post_treatment
from mayo_prostate.MRIcleaning.MRIcleaning import *

def extract_new_preMRIstatus(MRI_spreadsheet):
    # focus on the report data
    try:
        full = pd.read_csv(MRI_spreadsheet)
    except:
        full = pd.read_excel(MRI_spreadsheet)
    if "report_text" in full.columns:
        ds_report = pd.read_csv(MRI_csv, usecols=["report_text"])["report_text"]
        ds_report.name = "report"
    else:
        ds_report  = pd.read_csv(MRI_csv, usecols=["report"])["report"]

    # 1. pre-processing
    ds_report = MRIcleaning_preprocessing(ds_report)
    print("- Preprocessing: done")

    # 2. Split report into large section
    results, list_LARGESECTION_split_failed = MRIcleaning_split_into_large_section(ds_report)
    print("- Split report into large sections: done")

    # 3. Split "PROSTATE" section into smaller parts
    results, list_PROSTATE_split_failed, list_LOCALSTAGING_split_failed = MRIcleaning_split_prostate_section(results)
    print("- Split PROSTATE section into smaller parts: done")

    # 4. Add Impression PIRADS
    results = MRIcleaning_ImpPIRADS(results)
    print("- Extract Impression PIRADS: done")
    
    # 6. split lesion text into smaller parts.
    results, dict_LESION_split_failed = MRIcleaning_split_lesion_section(results)
    print("- Split lesion text into smaller parts: done")

    # 7. extract lesion-level information (location, PIRADS from overall category, exam-level PIRADS (based on lesion-level PIRADS and impression PIRADS), ADC, volume, length, T2-DWI-DCE scores, and zone category)
    results = MRIcleaning_extract_lesion_level_info(results)
    print("- Split lesion-level details : done")

    # 8. extract PreMRI_Gleason score, PSA values from "CLINICAL HISTORY"
    results["PreMRI_Gleason"] = results["CLINICAL_HISTORY"].apply(extract_PreMRI_worstGleason) #pre-MRI PIRADS
    results["PSA"] = results["CLINICAL_HISTORY"].apply(extract_PSA).apply(pd.Series)["NewestPSA"] #PSA
    print("- Extract PreMRI Gleason and PSA : done")

    # 10-1. extract pre-MRI Biopsy status by using "rule"
    results = MRIcleaning_rulebased_Bxstatus(results)

    # 11. assign pre-MRI status
    results =  MRIcleaning_preMRIstatus(results, ds_report)
    print("- Extract Pre-MRI status : done")

    # 10-2. Based on the pre-MRI status, update the rule-based BxStatus
    scr_bool = (results["Pre_MR_Dx"]=="Scr")
    # Not extracted -> Unknown Bx Status
    results.loc[scr_bool, "BxStatus_rule"] = results.loc[scr_bool, "BxStatus_rule"].fillna('Screening - Unknown Bx Status')
    # Not screening -> nan
    results.loc[~scr_bool, "BxStatus_rule"] = np.NaN
    print("- Extract BxStatus for screening pts (rule-based) : done")

    # 13. extract pre-MRI Biopsy status by using BERT model
    results = MRIcleaning_BERT_Bxstatus(results)
    print("- Extract BxStatus for screening pts (BERT model) : done")

    # 17. compare the Screening-BxStatus of DL and rules, if discrepancy existes -> manual check.
    scr = results[scr_bool]
    scr_same_Bx_index = scr[scr["BxStatus_DL"]==scr["BxStatus_rule"]].index
    scr_diff_Bx_index = scr[scr["BxStatus_DL"]!=scr["BxStatus_rule"]].index
    results.loc[scr_same_Bx_index, "Pre_MR_Dx"] = scr.loc[scr_same_Bx_index, "BxStatus_DL"]
    results.loc[scr_diff_Bx_index, "Pre_MR_Dx"] = "manual_check_BxStatus"
    print("- Comparison of Pre-MR Bx Status (DL vs Rule) : done")

    # 18. Assign manual check labels of pre-MRI Dx
    results["Pre_MR_Dx_wo_mc"] = results["Pre_MR_Dx"]
    pre_MR_dx_manual_bool = results["Pre_MR_Dx"].isin(["known PCa (details unknown)", "manual_check_BxStatus", "inappropriate", "manual_check_Hx_metastasis", "wo history"])
    results.loc[pre_MR_dx_manual_bool, "Pre_MR_Dx"] = "manual_check"
    print("- Flag manual check cases for Pre_MRI_Dx : done")

    # return a simple DataFrame 
    return_cols = ["exist_treatment_words", "Pre_MR_Dx_wo_mc", "Pre_MR_Dx"]
    return_cols_renamed = [f"new_{x}" for x in return_cols]
    return results[["report"]+return_cols].rename(columns:dict(zip(return_cols, return_cols_renamed)))
