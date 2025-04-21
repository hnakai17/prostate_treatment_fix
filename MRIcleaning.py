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

def main(MRI_csv):
    # focus on the report data
    full = pd.read_csv(MRI_csv)
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

    # 5. Add magnetic field strength
    results["magnet_strength"] = results["PROSTATE_MRI_TECHNIQUE"].apply(extract_magnetic_strength)
    print("- Extract Megntic filed strength: done")

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

    # 9. extract prostate volume from text
    results["PROSTATE_volume"] = results["PROSTATE_VOLUME"].apply(extract_volume) # "VOLUME": section, "volume": extracted volume
    print("- Extract Prostate volume : done")

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

    # 12. Flag how PIRADS was extracted. some reports need manual check of PIRADS scores.
    results = MRIcleaning_howPIRADSextracted(results)
    print("- PIRADS convertion : done")

    # 13. extract pre-MRI Biopsy status by using BERT model
    results = MRIcleaning_BERT_Bxstatus(results)
    print("- Extract BxStatus for screening pts (BERT model) : done")

    # convert the degree of confidence into 0-3.
    with open("local_ln_word2num.json", "r") as jsonfile:
        local_staging_LN_dic = json.load(jsonfile)

    # 14. extract local staging and conver the degree of confidence into 0-3
    results = MRIcleaning_local_staging(results, local_staging_cols = ['LOCAL_STAGING_CAPSULE', 'LOCAL_STAGING_NEUROVASCULAR_BUNDLE_INVASION', 'LOCAL_STAGING_SEMINAL_VESICLES_INVASION', 'LOCAL_STAGING_OTHER_ORGAN_INVASION'], local_staging_LN_dic=local_staging_LN_dic)
    print("- Connvert the degree of local staging : done")

    # 15. LN
    # Add the degree of suspicious of LN metastasis
    results = search_keywords_and_convert_into_score(results, "LYMPH_NODES", local_staging_LN_dic)
    print("- Connvert the degree of LN metastasis : done")
    # add LN location-length dataframe
    results["LYMPH_NODES_location_size_dic"] = results["LYMPH_NODES"].apply(extract_LN_main)
    print("- Extract the location and size of LN texts : done")

    # 16. Bone
    results = search_keywords_and_convert_into_score(results, "BONES", local_staging_LN_dic)
    print("- Connvert the degree of bone metastasis : done")

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

    # 19. Assign manual check labels of PSA, prostate volume
    mc_dic = {"PSA": 'PSA_mc', 'PROSTATE_volume': 'PROSTATE_volume_mc'}
    for col, new_col_mc in mc_dic.items():
        results[new_col_mc] = results[col].fillna("").str.contains("manual").replace(False, np.NaN)
        results.loc[results[col].fillna("").str.contains("manual").fillna(False), col] = np.NaN
    print("- Flag manual check cases for PSA and prostate volume : done")

    # 20. Sort columns
    results = MRIcleaning_sort_columns(results)

    # 21. concate another important columns
    need_cols = ["mrn", "date", "acc", "facility", "name", "age", "rad"]
    results = pd.concat([full[need_cols], results], axis=1)
    print("- Sort columns and add other important columns : done")

    return results, list_LARGESECTION_split_failed, list_PROSTATE_split_failed, list_LOCALSTAGING_split_failed, dict_LESION_split_failed

###########################################################################################################################################################################

def MRIcleaning_preprocessing(ds_report):
    # 1. pre-processing
    ds_report = ds_report.fillna("")
    for before in ["	+", "\t", "\r\n", "\n", r"\s+"]:
        ds_report = ds_report.apply(lambda text:re.sub(before, " ", text))
    ds_report = ds_report.apply(lambda text:re.sub("\s+:", ":", text))
    ds_report = ds_report.apply(convert_Newline_code)
    ds_report = ds_report.apply(assign_lesion_num) # assign lesion num
    ds_report = ds_report.replace("", np.NaN)
    return ds_report

def MRIcleaning_split_into_large_section(ds_report):
    # 2. Split report into large section
    """
    :param ds_report: Pandas Series after preprocessing
    :return: Pandas DataFrame with new following columns. EXAMS, CLINICAL HISTORY, COMPARISON, PROSTATE, PROSTATE BED, LOCAL STAGING, LYMPH NODES, BONES, OTHER FINDINGS, PROSTATE MRI TECHNIQUE, IMPRESSION, FINDINGS
    """
    section_list = ['EXAMS?:', 'CLINICAL\sHISTORY:', 'COMPARISON:', 'PROSTATE:', 'PROSTATE BED:','LOCAL STAGING:', 'LYMPH NODES:', 'BONES:', 'OTHER FINDINGS:', 'PROSTATE\sMRI\sTECHNIQUE:', 'IMPRESSION:', '(?<!OTHER\s)FINDINGS:']
    before_replace = ['LYMPH NODE:', 'BONE:', 'FINDING:', "History:", "(?<!TREATMENT\s)HISTORY:", "INFORMATION:", "TECHNIQUE:", 'PROSTATECTOMY BED:', 'Bone metastasis:', "\r\n", "Additional findings:", "Technique:", "Staging:", "Extraprostatic assessment:", "EXTRAPROSTATIC ASSESSMENT:"]
    after_replace = ['LYMPH NODES:', 'BONES:', 'FINDINGS:', "CLINICAL HISTORY:", "CLINICAL HISTORY:", "CLINICAL HISTORY:", "PROSTATE MRI TECHNIQUE:",'PROSTATE BED:', 'BONES:', "\n", 'OTHER FINDINGS:', "PROSTATE MRI TECHNIQUE:", 'LOCAL STAGING:', 'LOCAL STAGING:', 'LOCAL STAGING:']
    section_convert_dic = dict(zip(before_replace, after_replace))
    list_split_failed_text = []
    history_convert_index = ds_report.loc[(ds_report.notnull()) & (ds_report.fillna("").str.contains("History of")) & (~((ds_report.fillna("").str.contains("History:"))|(ds_report.fillna("").str.contains("HISTORY:"))))].index
    ds_report.loc[history_convert_index] = ds_report.loc[history_convert_index].apply(lambda text: re.sub("History of", "CLINICAL HISTORY:", text))
    results = ds_report.apply(lambda text: split_txt_into_section(text, section_list, list_split_failed_text=list_split_failed_text, section_convert_dic=section_convert_dic)).apply(pd.Series)
    results = pd.concat([ds_report, results], axis=1)
    return results, list_split_failed_text

def MRIcleaning_split_prostate_section(results):
    # 3. Split "PROSTATE" section into smaller parts
    """
    :param results: Pandas DataFrame with "PROSTATE" column
    :return: Pandas DataFrame with new following columns. EXAMS, CLINICAL HISTORY, COMPARISON, PROSTATE, PROSTATE BED, LOCAL STAGING, LYMPH NODES, BONES, OTHER FINDINGS, PROSTATE MRI TECHNIQUE, IMPRESSION, FINDINGS
    """
    # clean Lesion name (example,  Lesion # 1: -> Lesion1)
    results["PROSTATE"] = results["PROSTATE"].apply(convert_lesion)

    # Add Volume, Exam quality, PZ, TZ, and Lesion columns
    prostate_secs = ['Volume:', 'Exam quality:', 'Peripheral zone:', 'Transition zone:', "Lesion#\d"]
    list_split_failed_text1 = []
    tmp = results["PROSTATE"].apply(lambda text: split_txt_into_section(text, prostate_secs, list_split_failed_text=list_split_failed_text1)).apply(pd.Series)
    tmp.columns = [f"PROSTATE_{col}" for col in tmp.columns]
    results = pd.concat([results, tmp], axis=1)

    # cleaning column names of local invasion
    before_replace = ["Seminal vesicle invasion:", "Neurovascular bundle:", "Neurovascular bundles:", "Seminal vesicles:", "Seminal vesicle:"]
    after_replace = ["Seminal vesicles invasion:", "Neurovascular bundle invasion:", "Neurovascular bundle invasion:", "Seminal vesicles invasion:", "Seminal vesicles invasion:"]
    local_staging_dic = dict(zip(before_replace, after_replace))
    local_staging_secs = ["Capsule:", "Neurovascular bundle invasion:", "Seminal vesicles invasion:", "Other organ invasion:"]

    # Add new columns of local invasion (capsule, neurovascular bundle, SV, other organ) from "LOCAL STAGING" column
    list_split_failed_text2 = []
    tmp = results["LOCAL_STAGING"].apply(lambda text: split_txt_into_section(text, local_staging_secs, list_split_failed_text=list_split_failed_text2, section_convert_dic=local_staging_dic)).apply(pd.Series)
    tmp.columns = [f"LOCAL_STAGING_{col}" for col in tmp.columns]
    results = pd.concat([results, tmp], axis=1)
    return results, list_split_failed_text1, list_split_failed_text2

def MRIcleaning_ImpPIRADS(results):
    # 4. Add columns of Impression PI-RADS
    """
    :param results: Pandas DataFrame with "IMPRESSION" column
    :return: Pandas DataFrame with new columns of "IMPRESSION_PIRADS" (list) and "IMPRESSION_worst_PIRADS"
    """
    results["IMPRESSION_PIRADS"] = results["IMPRESSION"].apply(extract_PIRADS)
    results["IMPRESSION_worst_PIRADS"] = results["IMPRESSION_PIRADS"].apply(max_if_exist)
    # no PIRADS 3-5 -> worst PIRADS = 2
    no_PIRADS35_index = (results["IMPRESSION"].str.contains("no\sPI.?RADS.?3", case=False)) & (results["IMPRESSION_worst_PIRADS"].isnull())
    results.loc[no_PIRADS35_index, "IMPRESSION_worst_PIRADS"] = 2
    return results

def MRIcleaning_split_lesion_section(results):
    # 6. Split the lesion-level sentence
    """
    :param results: Pandas DataFrame with "LESION#" column
    :return: Pandas DataFrame with new columns of lesion-level size, zone, location, T2WI, DWI, DCE, Overall category, etc.
    """
    lesion_detail_list = []
    split_failed_dict = {}
    for Lesion_col in [col for col in results.columns if "LESION#" in col]:
        list_split_failed_text = []
        tmp = results[Lesion_col].apply(lambda text: split_txt_into_section(text, ["Size:", "Zone:", "Location:", "Imaging features:", "T2WI:", "DWI:", "DCE:", "Overall category:"], list_split_failed_text)).apply(pd.Series)
        tmp.columns = [f"{Lesion_col}_{x}" for x in tmp.columns]
        lesion_detail_list.append(tmp)
        split_failed_dict[Lesion_col] = list_split_failed_text
    results = pd.concat([results, pd.concat(lesion_detail_list, axis=1)], axis=1)
    return results, split_failed_dict

def MRIcleaning_extract_lesion_level_info(results):
    """
    extract lesion-level information of PIRADS score, Location, ADC, volume, size, T2-DWI-DCE scores, and zone.
    All consider the exam-level PIRADS score with considering "IMPRESSION PIRADS"
    :param results: Pandas DataFrame after "MRIcleaning_split_lesion_section"
    :return:
    """
    # 1.2: extract location from LOCATION column, and extract PIRADS from OVERALL_CATEGOR
    for keyword, function in zip(["LOCATION", "OVERALL_CATEGORY"], [extract_position, extract_PIRADS]):
        keyword_cols = [col for col in results.columns if keyword in col]
        tmp = pd.concat([results[col].apply(function) for col in keyword_cols], axis=1)
        tmp.columns = [col.replace(keyword, keyword.lower()) for col in keyword_cols]
        # try to extract unique PIRADS from the list. if no or more than PIRADS scores, need manual check.
        if keyword == "OVERALL_CATEGORY":
            for col in tmp.columns:
                notnull_index = tmp[tmp[col].notnull()].index
                tmp.loc[notnull_index, col] = tmp.loc[notnull_index, col].apply(extract_max_PIRADS)
        results = pd.concat([results, tmp], axis=1)

    # extract-PIRADS score from each lesion's overall_category
    PIRADS_col = [col for col in results.columns if ("overall_category" in col) or ("IMPRESSION_worst_PIRADS" in col)]
    # assign exam-level PIRADS
    results["exam_level_PIRADS"] = results[PIRADS_col].fillna(0).max(axis=1).replace(0, np.NaN)

    # 3. Extract ADC from lesion-level DWI text
    DWI_columns = [col for col in results.columns if "DWI" in col]
    for DWI_col in DWI_columns:
        results[f"{DWI_col[:-3]}ADCvalue"] = results[DWI_col].apply(extract_ADC)

    ## 4. extract size and volume for each lesion

    # volume
    size_cols = [col for col in results.columns if "SIZE" in col]
    for col in size_cols:
        results[f"{col}_volume"] = results[col].apply(extract_volume)

    # length
    for col in size_cols:
        results[f"{col}_cm"] = results[col].apply(extract_cm)

    # maximum length
    for col in size_cols:
        results[f"{col}_max_cm"] = results[f"{col}_cm"].apply(max_if_exist)

    ## 5. extract T2 and DWI score for each lesion.
    T2_DWI_cols = [x for x in results.columns if ("T2WI" in x) or ("DWI" in x)]
    for col in T2_DWI_cols:
        results[f"{col}score"] = results[col].apply(extract_T2_DWI_score)

    ## 6. extract DCE score for each lesion. (if contain "positive" -> 1.)
    DCE_cols = [x for x in results.columns if "DCE" in x]
    for col in DCE_cols:
        results[f"{col}score"] = results[col].fillna("").str.contains("positive", case=False).astype(int)
        results.loc[results[col].isnull(), f"{col}score"] = np.NaN

    ## 7. extract ZONE category for each lesion. (if contain "peripheral", "P". if contain "transition", "T". If contains both, "TP")
    LESION_ZONE_cols = [x for x in results.columns if (("ZONE" in x) and ("LESION" in x))]
    for col in LESION_ZONE_cols:
        contain_p = results[col].fillna("").str.contains("peripheral", case=False)
        contain_t = results[col].fillna("").str.contains("transition", case=False)
        results.loc[(contain_p & contain_t), f"{col}_cat"] = "TP"
        results.loc[(~contain_p & contain_t), f"{col}_cat"] = "T"
        results.loc[(contain_p & ~contain_t), f"{col}_cat"] = "P"
        results.loc[results[col].isnull(), f"{col}_cat"] = np.NaN
    return results


def MRIcleaning_rulebased_Bxstatus(results, patten_jsonfile = "Bx_phrase.json"):
    """
    :param resutls: Add BxStatus from "CLINICAL HISTORY"
    :param patten_jsonfile: Pattern created by previous work
    :return: new dataframe. "BxStatus_rule" contains new category. It will be compared later with DL one.
    """
    results[['Screening - Negative', 'Screening - Naive', 'Screening - Unknown Bx Status']] = results["CLINICAL_HISTORY"].apply(lambda txt:extract_Bx_status(txt, patten_jsonfile)).apply(pd.Series)
    # summary the Bx status results in "Bx_status"
    for col in ['Screening - Negative', 'Screening - Naive', 'Screening - Unknown Bx Status']:
        results.loc[results[col], "BxStatus_rule"] = col
    results.loc[results[['Screening - Negative', 'Screening - Naive', 'Screening - Unknown Bx Status']].sum(axis=1)>1, "BxStatus_rule"] = "manual_check"
    return results


def MRIcleaning_preMRIstatus(results, ds_report, preMRIstatus_jsonfile = "preMRI_status.json"):
    """
    1. first, check if some keywords (saved in the JSON file) exist in the texts.
        -> "exist_Scr_words", "exist_AS_words", "exist_treatment_words", "exist_incomplete_words", "exist_research_word" columns
            - Scr, AS related keywords: from CLINICAL HISTORY
            - treatment, incomplete, and research keywords: from whole text
    :param results: Pandas DataFrame
    :param ds_report: preprocessed DataSeries of report text
    :param preMRIstatus_jsonfile: Json file containing text keywords for each sentence.
    :return: updated DataFrame
    "Pre_MR_Dx":
    1. "Scr": "exist_Scr_words" or PreMRI_Gleaon<6
    2. "GS6": "exist_AS_words" or PreMRI_Gleaon=6
    3. "csPCa" (or with treatment Hx): some keywords (prostatectomy, prostate bed etc) exist or PreMRI_Gleason>=7
    4. "manual_check_Hx_metastasis": metastasis exists in the CLINICAL HISTORY. check if it is prostate-related or not.
    5. "known PCa (details unknown)": needs manually check. it there is no detailed information, it will be removed from the study
    """
    # load keywords of screening, AS, treatment, incomplete-MRI, and research
    with open(preMRIstatus_jsonfile, "r") as preMRI_status:
        preMRI_status_dic = json.load(preMRI_status)

    # extract keywords of screening, AS, treatment, incomplete-MRI, and research
    results["exist_Scr_words"] = (results["CLINICAL_HISTORY"].str.contains(preMRI_status_dic["Scr"], case=False)|results['Screening - Naive']).fillna(False)
    results["exist_AS_words"] = results["CLINICAL_HISTORY"].str.contains(preMRI_status_dic["AS"], case=True).fillna(False)
    results["exist_treatment_words"] = ds_report.str.contains(preMRI_status_dic["treatment_related"], case=False).fillna(False) #results["CLINICAL_HISTORY"].str.contains(preMRI_status_dic["treatment_related"], case=False).fillna(False)
    results["exist_incomplete_words"] = ds_report.str.contains(preMRI_status_dic["incomplete"], case=False).fillna(False)
    results["exist_reseach_word"] = ds_report.str.contains("research", case=False)

    # The following few lines of code were added in April 2025 to update the post-treatment status for csPCa.
    results["is_post_treatment"] = ds_report.apply(is_post_treatment)
    results["exist_treatment_words"] = results["exist_treatment_words"] & results["is_post_treatment"]

    # consider the status in combination (with Gleason score etc.)
    preMRI_scr = results["exist_Scr_words"]|results["PreMRI_Gleason"].isin([4,5])
    preMRI_AS = results["exist_AS_words"]|results["PreMRI_Gleason"].isin([6])
    preMRI_inappropriate = results[["exist_incomplete_words", "exist_reseach_word"]].sum(axis=1)>=1

    # The following few lines of code were added in April 2025 to update the post-treatment status for csPCa.
    ## previous code
    ### preMRI_csPCa = ds_report.str.contains("prostatectomy", case=False)|results["CLINICAL_HISTORY"].fillna("").str.contains("metastatic prostate (cancer|carcinoma)|prostate (cancer|carcinoma) with bone metastas", case=False)|results["exist_treatment_words"]|results["PreMRI_Gleason"].isin([7,8,9,10])|results["IMPRESSION"].str.contains("radiation|planning|carbon seed|carbon marker|fiducial marker", case=False).fillna(False)|results["FINDINGS"].str.contains("radiation|planning|carbon seed|carbon marker|fiducial marker", case=False).fillna(False)|results["PROSTATE_BED"].notnull()
    ## new code using updated post-treatment bool. 
    cond_meta = results["CLINICAL_HISTORY"].fillna("").str.contains("metastatic prostate (cancer|carcinoma)|prostate (cancer|carcinoma) with bone metastas", case=False)
    cond_post_treatment = results["exist_treatment_words"]
    cond_preMRIGleason_over7 = results["PreMRI_Gleason"].isin([7,8,9,10])
    cond_with_prostatebed = results["PROSTATE_BED"].notnull()
    preMRI_csPCa = cond_meta|cond_post_treatment|cond_preMRIGleason_over7|cond_with_prostatebed

    # assign pre-MRI status
    results.loc[preMRI_scr, "Pre_MR_Dx"] = "Scr"
    results.loc[preMRI_AS, "Pre_MR_Dx"] = "GS6"
    results.loc[preMRI_csPCa, "Pre_MR_Dx"] = "csPCa" # or with treatment history
    results.loc[preMRI_inappropriate, "Pre_MR_Dx"] = "inappropriate"

    # containing metastasis in CLINICAL HISTORY and not csPCa -> manual check
    susp = (results[(results["CLINICAL_HISTORY"].fillna("").str.contains("metasta", case=False))]["Pre_MR_Dx"]!="csPCa")
    results.loc[susp[susp].index, "Pre_MR_Dx"] = "manual_check_Hx_metastasis"

    # Missing pre-MRI status -> most of the cases, "Scr". sometimes, prostate cancer wo details.
    def classify_unknown_status_with_cancer(texts):
        check_text_list = [text for text in texts.split(". ") if ("prostate (adeno)?carcinoma" in text.lower()) or ("prostate cancer" in text.lower()) or ("malignan" in text.lower())]
        scr_list = [text for text in check_text_list if re.search("predisposition|exclude|suspected|suspicious|(without|no) known|evaluate|concern|(no|without) (personal )?history", text, re.IGNORECASE)]
        if len(scr_list)>=1:
            return "Scr"
        else:
            return "known PCa (details unknown)"

    # if containing cancer-related keywords, check if it is screening or not. -> "Scr" or "known cancer"
    unknown_status_with_cancer = results["CLINICAL_HISTORY"][results["Pre_MR_Dx"].isnull()&
                                                             results["CLINICAL_HISTORY"].fillna("").str.contains("prostate cancer|prostate (adeno)?carcinoma|malignan", case=False)]
    results.loc[unknown_status_with_cancer.index, "Pre_MR_Dx"] = unknown_status_with_cancer.apply(classify_unknown_status_with_cancer)

    # if still unclear pre-MRI status, consider it as "screening"
    results.loc[(results["CLINICAL_HISTORY"].notnull() & results["Pre_MR_Dx"].isnull()), "Pre_MR_Dx"] = "Scr"
    results.loc[(results["CLINICAL_HISTORY"].isnull() & results["Pre_MR_Dx"].isnull()), "Pre_MR_Dx"] = "wo history"
    return results

def MRIcleaning_howPIRADSextracted(results):
    """
    Flag how PIRADS was extracted. (Some reports need manual check because PIRADS score was not extracted. Report mentioning no suspicious lesion, but without PIRADS scores will be assigned PIRADS 2 automatically with "converted" flag. )
    :param results:
    :return:
    """
    # if exam-level-PIRADS exist -> "extracted"
    results.loc[results["exam_level_PIRADS"].notnull(), "PIRADS_extraction"] = "extracted"

    # No PIRADS score in spite of the screening of AS status
    scr_As_bool = results["Pre_MR_Dx"].str.contains("Scr|GS6")
    no_PIRADS_bool = results["exam_level_PIRADS"].isnull()
    scr_AS_noPIRADS = results.loc[(scr_As_bool & no_PIRADS_bool), "IMPRESSION"]
    # Assign PIRADS 2 if some text suggesting no clinically significant cancer exist in the "IMPRESSION" section.
    regex_list_suggesting_no_susp_lesion = ["clinically significant prostate (carcinoma|malignancy|cancer) is unlikely to be present",
                                            "((?:No |Nothing )(?:[^\.])*(?:concerning|worrisome|significant|intermediate|high grade|high-grade|focal|prostat|suspicio|PIRADS.*5)(?:[^\.])*(?:carcinoma|malignancy|cancer|lesion|tumor|neoplasm))"]
    regex_suggesting_no_susp_lesion = "|".join(regex_list_suggesting_no_susp_lesion)
    # keywords suggesting no suspicious lesion in screening or AS exams
    scr_AS_noPIRADS_text_suggesting_no_susp_lesions = scr_AS_noPIRADS.fillna("").apply(lambda text:re.findall(regex_suggesting_no_susp_lesion, text, re.IGNORECASE))

    def remove_some_keywords_from_sentence(list_of_tuple_containing_sentence, remove_keywords_list):
        lis = []
        remove_keywords_regex = "|".join(remove_keywords_list)
        for tuple_containing_sentence in list_of_tuple_containing_sentence:
            for sentence in tuple_containing_sentence:
                if (len(sentence)<2) or (re.search(remove_keywords_regex, sentence)):
                    pass
                else:
                    lis.append(sentence)
        return lis

    # if the following keywords exists, remove that sentence from the converting list.
    remove_keywords_list = ["new", "bone", "residual", "recurrence"]
    scr_AS_noPIRADS_text_suggesting_no_susp_lesions_after_removing_keywords = scr_AS_noPIRADS_text_suggesting_no_susp_lesions.apply(lambda text:remove_some_keywords_from_sentence(text, remove_keywords_list))
    PIRADS_convert_index = scr_AS_noPIRADS_text_suggesting_no_susp_lesions_after_removing_keywords[scr_AS_noPIRADS_text_suggesting_no_susp_lesions_after_removing_keywords.apply(len)>=1].index
    results.loc[PIRADS_convert_index, "PIRADS_extraction"] = "converted"
    results.loc[PIRADS_convert_index, "IMPRESSION_worst_PIRADS"] = 2
    results.loc[PIRADS_convert_index, "exam_level_PIRADS"] = 2
    #results.loc[PIRADS_convert_index, "exam_level_PIRADS"] = results.loc[PIRADS_convert_index, PIRADS_col].fillna(0).max(axis=1).replace(0, np.NaN)

    # no "extracted" or "converted" -> manual check
    results.loc[scr_As_bool, "PIRADS_extraction"] = results.loc[scr_As_bool, "PIRADS_extraction"].fillna("manual_check")
    return results

def MRIcleaning_BERT_Bxstatus(results, trained_weight="/research/projects/Nakai/Report_cleaning/Models/models/MRI_BxStatus_20230615.pt"):
    """
    :param results:
    :param trained_weight: Hugging Face BERT 3-class classification model. Population: Screening.
    :return: updated dataframe with prediction and probability
    """
    label_category_dic = {0:'Screening - Naive', 1:'Screening - Negative', 2:'Screening - Unknown Bx Status'}
    num_labels = len(label_category_dic.keys())
    pretrained = "bert-base-cased"
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 16

    ## Model setting ##
    model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=num_labels)
    weight = torch.load(trained_weight)
    
    # added May 16 2024 because of the following error. Error(s) in loading state_dict for BertForSequenceClassification: Unexpected key(s) in state_dict: "bert.embeddings.position_ids".
    # SEE https://github.com/huggingface/transformers/issues/24921
    del weight["bert.embeddings.position_ids"]
    

    model.load_state_dict(weight)
    model.to(device)

    scr_bool = (results["Pre_MR_Dx"]=="Scr")
    scr_Hx = results[scr_bool][["CLINICAL_HISTORY"]].rename(columns={"CLINICAL_HISTORY":"CONTAINING SENTENCE"}).fillna("")
    dataloader = nlp.prepare_dataloader(scr_Hx, batch_size=batch_size, shuffle=False)
    model.eval()
    model_results_dic = nlp.apply_model(dataloader, model)
    predicted_prob_all, predictions_all = model_results_dic["prob"], model_results_dic["pred"]

    # 1. add pred
    results.loc[scr_Hx.index, "BxStatus_DL"] = pd.Series(torch.cat([x.cpu() for x in predictions_all]).numpy(), name="pred").values

    # 2. add prob
    results.loc[scr_Hx.index, [f"BxStatus_DLprob_{label_category_dic[i]}" for i in range(num_labels)]] = pd.DataFrame(torch.cat([x.cpu() for x in predicted_prob_all]).numpy()).values
    results.loc[scr_Hx.index, "BxStatus_DL"] = results["BxStatus_DL"].replace(label_category_dic)
    return results

def MRIcleaning_local_staging(results,  local_staging_cols, local_staging_LN_dic):
    #local_staging_cols = ['LOCAL_STAGING_CAPSULE', 'LOCAL_STAGING_NEUROVASCULAR_BUNDLE_INVASION', 'LOCAL_STAGING_SEMINAL_VESICLES_INVASION', 'LOCAL_STAGING_OTHER_ORGAN_INVASION']

    # Add local staging columns of extracted keywords and converted integer (0: negative -> 3: positive)
    for local_staging_col in local_staging_cols:
        results = search_keywords_and_convert_into_score(results, local_staging_col, local_staging_LN_dic)
    return results

def MRIcleaning_sort_columns(results):
    # Sort columns: 'report', 'CLINICAL_HISTORY', 'IMPRESSION', 'FINDINGS', 'magnet_strength', 'Pre-MRI info',
    # 'BxStatus', 'PIRADS', 'PSA', 'Lesion level', 'TNM', 'COMPARISON', 'EXAM', 'OTHER_FINDINGS'
    def extract_sorted_cols(word, bans=None):
        collist_containing_word = sorted([x for x in cols if word in x])
        if bans:
            return [x for x in collist_containing_word if bans not in x]
        else:
            return collist_containing_word

    cols = results.columns
    lis = []
    for word in ["Pre", "BxStatus", "exist", "PIRADS", "PSA"]:
        lis.append(extract_sorted_cols(word))
    lis.append(extract_sorted_cols("PROSTATE", "LESION"))
    for word in ["LESION", "LOCAL_STAGING", "LYMPH_NODE", "BONES"]:
        lis.append(extract_sorted_cols(word))

    extracted_cols = [x for sublis in lis for x in sublis]
    sorted_cols = ['report', 'CLINICAL_HISTORY', 'IMPRESSION', 'FINDINGS', 'magnet_strength'] + extracted_cols + ['COMPARISON', 'EXAM', 'OTHER_FINDINGS']
    return results[sorted_cols]

###########################################################################################################################################################################


def convert_Newline_code(text):
    try:
        text = re.sub("_x000D_", " ", text)
        return text
    except:
        return text

def assign_lesion_num(text):
    """
    Assign lesion numbers. (Because some case do not have lesion number)
    :param text:
    :return: updated text
    Example
        Lesion # 1,,,, Lesion # ,,,,,,, -> Lesion # 1,,,, Lesion #2,,,,,,,
    """
    lesion_match_list = [match for match in re.finditer("Lesion\s?#\s?\d?:?", text)]
    for i, lesion_match in enumerate(lesion_match_list, 1):
        span = lesion_match.span()
        text = text[:(span[1]-1)] + str(i) + text[span[1]:]
    return text

def split_txt_into_section(rep_txt, section_list, list_split_failed_text=None, section_convert_dic=None):
    """
    Split txt into smaller sections.
    :param rep_txt: txt
    :param section_list: list of section name
    :param section_convert_dic: dictionary to uniform text description. Change the "key" into "value".
    :return: dictionary (key: each section name, value: section contents)
    """
    try:  # in case some section exist
        # replace some keywords (before_replacement -> after_replacement)
        rep_txt += " "
        if section_convert_dic:
            for b in section_convert_dic.keys():
                a = section_convert_dic[b]
                rep_txt = re.sub(b, a, rep_txt)
                #compiled = re.compile(re.escape(b), re.IGNORECASE)
                #rep_txt = compiled.sub(a, rep_txt)
        # split into section
        section_list = "|".join(section_list)
        section_list = [key.strip("\n") for key in re.findall(f"(?:{section_list})", rep_txt, re.IGNORECASE)] # (?:\:|#\s?\d)

        # extract the index of the section
        key_spans = [(key, re.search(key, rep_txt).span()) for key in section_list] #  if key not in ["Size:", "Zone:", "Location:", "T2WI:", "DWI:", "DCE:", "Overall category:"]
        key_index_df = pd.DataFrame(dict(key_spans)).T
        key_index_df.columns=["content_end", "content_start"]
        key_index_df["content_end"].iloc[:-1] = key_index_df["content_end"].iloc[1:]
        key_index_df["content_end"].iloc[-1] = -1
        key_index_df["content"] = key_index_df.apply(lambda x:rep_txt[x["content_start"]:x["content_end"]], axis=1)
        key_index_df["content"] = key_index_df["content"].str.replace("\n", " ")

        # remove ":" from the section
        key_index_df.index = [x.replace(":", "").replace(" ", "_").upper() for x in key_index_df.index]

        section_dict = key_index_df["content"].to_dict()
        return section_dict
    except: # in case where no section exist (print the text for a reference)
        if pd.notnull(rep_txt):
            if type(list_split_failed_text)==list:
                list_split_failed_text.append(rep_txt)

def convert_lesion(text):
    """
    Uniform the lesion description.
    Example:
        "Lesion # 1:" -> "Lesion#1"
    """
    if pd.notnull(text):
        return re.sub("Lesion\s?#\s?(\d):?", "Lesion#\\1", text)
    else:
        return text

def findall_mod(text, regex):
    if pd.notnull(text):
        return re.findall(regex ,text)
    else:
        return text

############ Extract PIRADS ############
def extract_PIRADS(text):
    """
    Extract PIRADS scores from text and save them as list of integer. If the text is null, return null.
    :param text:
    :return: first, split the txt into each sentence. Then, extract clean PIRADS from each sentence and create list of clean PIRADS scores
    Example
        PIRADS 3 -> [3]
        PI-RADS score: 3 -> [3]
        no PIRADS 3-5 -> []
        PIRADS four -> [4]
        PIRADS V -> [5]
        # previous or downgrade -> only extract PIRADS with "to"
        previously described PIRADS 5 -> []
        downgraded to PIRADS 3 -> [3]
    """
    # normal pattern
    pattern = r"(?<!no\s)PI.?RADS.?\s?(?:category:?\s?|score:?\s?)?([1-5]{1}|II|III|IV|V(?!ersion|\s?2)|one|two|three|four|five)(?!-5|(?:,\s?|/)4(?:,\s?|/| or |,\s?or )5)"
    # when some keywords existed like previous|prior|downgrade, use PIRADS only with "to"
    pattern_non_past = r"(?<!no\s)to (?:a )?PI.?RADS.?\s?(?:category\s?|score\s?)?([1-5]{1}|II|III|IV|V(?!ersion|\s?2)|one|two|three|four|five)(?!-5|(?:,\s?|/)4(?:,\s?|/| or |,\s?or )5)"
    PIRADS_dic = {"II":2, "III":3, "IV":4, "V":5, "one":1, "two":2, "three":3, "four":4, "five":5}

    if pd.isnull(text):
        return text
    else:
        # split the text into each sentence. If the sentence begin with "No", we don't try to extract PIRADS from this sentence.
        sentence_list = text.split(". ") # not extracting float number like 1.8
        sentence_list = [x for sec in sentence_list for x in re.split(r"[@\(\)]", sec)]
        sentence_list_not_starting_with_no = [x for x in sentence_list if not re.match("^\s*no", x, re.IGNORECASE)]

        # store PIRADS score from each sentence.
        PIRADS_list = []
        for sentence in sentence_list_not_starting_with_no:
            previous_downgrade_keywords_exist = re.findall("previous|downgrade", sentence, re.IGNORECASE)
            if not previous_downgrade_keywords_exist:
                extracted_PIRADS = re.findall(pattern, sentence, re.IGNORECASE)
            else:
                extracted_PIRADS = re.findall(pattern_non_past, sentence, re.IGNORECASE)
            for PIRADS in extracted_PIRADS:
                PIRADS_list.append(PIRADS)

        # convert the PIRADS contents into int objects.
        def replace_(x):
            if len(x)==0:
                return x
            else:
                if x in PIRADS_dic.keys():
                    return PIRADS_dic[x]
                else:
                    return int(x)
        PIRADS_list = list(map(replace_, PIRADS_list))

        # return unique PIRADS
        unique_PIRADS_list = list(np.unique(PIRADS_list))
        return sorted(unique_PIRADS_list)

def extract_max_PIRADS(PIRADS_list):
    if len(PIRADS_list)==0:
        return np.NaN
    else:
        return np.max(PIRADS_list)

def max_if_exist(x):
    try:
        return max(x)
    except:
        return np.NaN

def extract_T2_DWI_score(T2_DWI_text):
    if pd.isnull(T2_DWI_text):
        return np.NaN
    else:
        lis = re.findall("score\s?(\d)", T2_DWI_text)
        if len(lis)==1:
            return lis[0]
        elif len(lis)==0:
            return np.NaN
        else:
            return "manual_check"

############ Extract PSA ############
# regex1: "<digit> (as of/on/in/) <date>"
paired_regex1 = r"(\d+\.?\d*),?\s(?:ng/ml\s)?(?:as\sof\s|on\s|in\s)\(?(\d{1,2}/\d{1,2}/\d{2}(?!\d)|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\w+\s(?:of\s)?\d{4}|\w+\s\d{2},?\s\d{4})\)?"
# regex2: "<date> (was/is/at/-) <digit>"
paired_regex2 = r"\(?(\d{1,2}/\d{1,2}/\d{2}(?!\d)|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\w+\s\d{4}|\w+\s\d{2},?\s\d{4})\)?\s?(?::|was|is|at|-)?\s\(?(\d+\.?\d*)\)?"
# regex3: minor function. (Paird Date, PSA like regex1, regex2.) Importance; regex1>regex2>>regex3.
paired_regex3 = r"\(?(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\w+\s\d{4}|\w+\s\d{2},?\s\d{4}),\s(\d+\.?\d+)(?:;|\.)\D"

def extract_paired_PSA(txt, paired_regex_list = [paired_regex1, paired_regex2, paired_regex3]):
    """
    :param txt:
    :param regex:
    :return: df with PSA, Date values (not depend on the order of PSA and Date)
    """
    if pd.isnull(txt):
        pass
    else:
        PSADate_list = []
        try:
            for paired_regex in paired_regex_list:
                RegexOutput = re.findall(paired_regex, txt, re.IGNORECASE)
                PSADate = pd.DataFrame(RegexOutput, columns=["Column1", "Column2"])
                # baned keywords
                banned = ["and", "in", "on", "at"]
                f = lambda x: ' '.join([item for item in x.split(" ") if item not in banned])
                PSADate["Column1"] = PSADate["Column1"].apply(f)
                PSADate["Column2"] = PSADate["Column2"].apply(f)
                try:
                    PSADate["Date"] = pd.to_datetime(PSADate["Column2"])
                    PSADate = PSADate.drop(columns = "Column2")
                    PSADate = PSADate.rename(columns={"Column1":"PSA"})
                except:
                    PSADate["Date"] = pd.to_datetime(PSADate["Column1"])
                    PSADate = PSADate.drop(columns = "Column1")
                    PSADate = PSADate.rename(columns={"Column2":"PSA"})
                PSADate_list.append(PSADate)
            df = pd.concat(PSADate_list).sort_values("Date", ascending=False)
            if len(df)>=1:
                NewestPSA = float(df.iloc[0]["PSA"])
                if NewestPSA>=2010:
                    NewestPSA = "manual_check"
                NewestDate = df.iloc[0]["Date"]
            else:
                NewestPSA = np.NaN
                NewestDate = np.NaN
            return {"NewestPSA":NewestPSA, "NewestPSA_Date":NewestDate} #df
        except:
            return {"NewestPSA":"manual_check", "NewestPSA_Date":"manual_check"}

# unpaired_regex1: "psa <digit>. (with period)" Unpaired PSA. Prioritize these values than regex1-3 results.
unpaired_regex = r"psa(?:\s*)?:?(?:\s*)?(?:\w+\s+){0,2}(?:=\s)?(\d+\.?\d*)(?!-year)(?:\s*)?(?:ng/ml|ng/mL)?"
most_recent_regex = "most recent " + unpaired_regex

def extract_unpaired_PSA(text):
    if pd.isnull(text):
        return {"NewestPSA":np.NaN}
    else:
        most_recent_PIRADS = re.findall(most_recent_regex, text, re.IGNORECASE)
        if len(most_recent_PIRADS)==1:
            return {"NewestPSA":float(most_recent_PIRADS[0])}
        else:
            unpaired_PSA_list = re.findall(unpaired_regex, text, re.IGNORECASE)
            unique_PSA_list = list(np.unique(unpaired_PSA_list))
            if len(unique_PSA_list)==1:
                PSA = float(unique_PSA_list[0])
                if PSA>=2010:
                    PSA = "manual_check"
                return {"NewestPSA":PSA}
            elif len(unique_PSA_list)>=2:
                return {"NewestPSA":"manual_check"}
                """
                unique_PSA_list.append("manual check")
                return {"NewestPSA":unique_PSA_list} 
                """
            else:
                return {"NewestPSA":np.NaN}

def convert_less_than_01PSA(text):
    if pd.isnull(text):
        return text
    else:
        text = re.sub("<\s?0\.10?\s?", "0.1", text)
        return re.sub("less than 0\.10?\s?", "0.1", text)

def extract_PSA(text):
    """
    Extract PSA
    :param text: text in Clinical history
    :return: float or "manual check" or np.NaN

    ## date: (01/31/1985; 01/1985; January 1985; January 31, 1985;)
    ## PSA: digit with/wo ng/ml.
    """
    if pd.isnull(text):
        return text
    else:
        text = convert_less_than_01PSA(text)
        unpaired_PSA = extract_unpaired_PSA(text)
        paired_PSA = extract_paired_PSA(text)
        PSA = unpaired_PSA
        if (type(unpaired_PSA["NewestPSA"])!=list):
            if pd.isnull(unpaired_PSA["NewestPSA"]):
                PSA = paired_PSA
                if pd.isnull(PSA["NewestPSA"]) & pd.notnull(text) & (len(re.findall("PSA", text))>=1) & (len(re.findall("\d", text))>=1):
                    PSA = {"NewestPSA":"manual_check"}
        try:
            if PSA["NewestPSA"]>2010:
                PSA["NewestPSA"] = "manual_check"
        except:
            pass
        return PSA

############ Extract Prostate volume ############
def extract_volume(text):
    """
    Clean volume from list. If there exist one candidate, extract that volume. Else, need manual check.
    If the text is null, return null.
    Example
        [80] -> 80
        [40, 22] -> "manual check"
    """
    volume = findall_mod(text, r"(\d+\.?\d*)\s?(?:cc|ml|mL)")
    try:
        if len(volume)==1:
            return volume[0]
        elif len(volume)>1:
            return "manual_check"
    except:
        return volume

############ Extract Prostate lesion length ############
def extract_cm(SIZE_text):
    if pd.isnull(SIZE_text):
        return np.NaN
    else:
        string_list =  re.findall("(\d+\.?\d*)\s?(?:x|cm|mm)", SIZE_text)
        if re.search("(?:\d)\s?mm", SIZE_text, re.IGNORECASE):
            return [0.1 * float(x) for x in string_list]
        elif re.search("(?:\d)\s?cm", SIZE_text, re.IGNORECASE):
            return [float(x) for x in string_list]
        else:
            return np.NaN

############ Extract position ############
#### Variables for extracting positions (below function) ####
RL = ["Right", "Left"] #"left", "right"
AP = ["Anter", "Poster"] # ("A", "P")
LM = ["Lateral", "Media|Midline"] #"lateral", "media"
BMA = ["Base", "Mid", "Apex"] #"base", "midgland|mid", "apex"
location_list = ["".join(x) for x in product([y[0] for y in RL],
                                             [y[0] for y in AP],
                                             [y[0] for y in LM],
                                             [y[0] for y in BMA])]
# After creating location_list, change the element not to capture similar words
LM[0] = "(?<!Bi)Lateral"
LM[1] = "(Media|Midline)"
BMA[1] = "Mid(?!line)"

def extract_position(text, location_list=location_list):
    """
    Extract position keyword from Each lesion's location text. If the text is null, return null.
    :param text:
    :param location_list: list containing 24 positioning text
    :return: list containing positioning text of the lesion
        text[0]: R or L (right or left)
        text[1]: A or P (anterior or posterior)
        text[2]: L or M (lateral or median)
        text[3]: B or M or A (base or mid-grand or apex)
    Example
        "Left posterior lateral at base" -> ['LPLB']
        "Right posterior medial at apex to base" -> ['RPMB', 'RPMM', 'RPMA']
        "Midline posterior at base with extension to the apex." -> ['RPMB', 'RPMM', 'RPMA', 'LPMB', 'LPMM', 'LPMA']
        "Predominantly left involving the near-entirety of the gland (apex to base)"
        -> ['LALB', 'LALM', 'LALA', 'LAMB', 'LAMM', 'LAMA', 'LPLB', 'LPLM', 'LPLA', 'LPMB', 'LPMM', 'LPMA']
    """
    if pd.isnull(text):
        return text
    else:
        # if text contain both apex and base, add mid (to capture mid)
        if (len(re.findall("apex", text, re.IGNORECASE))>=1) & (len(re.findall("base", text, re.IGNORECASE))>=1):
            text += " Mid"
        # list to return
        location_in_text = location_list.copy()
        for i, lis in enumerate([RL, AP, LM, BMA]):
            # if all element do not exist, pass (if neither right nor left existed, contain both right and left)
            if not re.search("|".join(lis), text, re.IGNORECASE):
                pass
            else:
                # bilateral -> capture right and left
                if (lis==RL) & (len(re.findall("bilateral", text, re.IGNORECASE))>=1):
                    pass
                else:
                    # check if each keyword exist in the text. if not existed, delete that direction from the "location_in_text"
                    for keyword in lis:
                        if not re.search(keyword, text, re.IGNORECASE):
                            if lis != LM:
                                location_in_text = [x for x in location_in_text if x[i] != keyword[0]]
                            else:
                                location_in_text = [x for x in location_in_text if x[i] != keyword[7]] # Each seventh word in LM is "L" or "M" (just a coincidence)
        return location_in_text

############ Bx status ############
def extract_Bx_status(Hx_txts, patten_jsonfile):
    """
    In the future, also consider the results of pre-MRI pathology reports
    :param Hx_txts: multiple sentences. (especially in the MRI history section)
    :param patten_jsonfile: json files containing Bx patten. Originally created by Drs. Nagayama, Takahashi
    :return: dictionary
    """
    # Negative_idx_path = df['Pre_MR_Dx_Path_Report'].str.contains('Benign - Biopsy', na=False)
    # df['Negative_idx_path'] = Negative_idx_path
    # df.loc[Negative_idx_path, 'Bx_Status'] = 'Negative'

    with open(patten_jsonfile, "r") as f:
        patten_dic = json.load(f)

    if pd.isnull(Hx_txts):
        Hx_txts = ""
    def idx_phrase(txt, *pattern):
        """
        :param txt:
        :param pattern: values in patten_dic
        :return: check if all pattens exist in the txt.
        """
        return all([(re.search(p, txt, re.IGNORECASE)) for p in pattern])

    negative_list, naive_list, unknown_list = [],[],[]
    negative_list.append(('screening - history of benign biopsy' in Hx_txts.lower()))
    naive_list.append(('screening - biopsy naive' in Hx_txts.lower()))
    unknown_list.append(('screening - biopsy status unknown' in Hx_txts.lower()))

    for Hx_txt in Hx_txts.split(". "):
        negativecomb12 = idx_phrase(Hx_txt, patten_dic["Negative_comb1"], patten_dic["Negative_comb2"])
        negativecomb234 = idx_phrase(Hx_txt, patten_dic["Negative_comb2"], patten_dic["Negative_comb3"], patten_dic["Negative_comb4"])
        negativecomb56 = idx_phrase(Hx_txt, patten_dic["Negative_comb5"], patten_dic["Negative_comb6"])
        negative_phrase = negativecomb12 or negativecomb234 or negativecomb56
        naive_phrase = idx_phrase(Hx_txt, patten_dic["Naive_comb1"], patten_dic["Naive_comb2"])
        negative_list.append(negative_phrase)
        naive_list.append(naive_phrase)
    return {"Negative":any(negative_list), "Naive":any(naive_list), "Unknown":any(unknown_list)}


############ Pre-MRI Gleason score ############
regex_Gleason_with_plus = "(\d|for)\s?\+\s?(\d|for)" # 4+4, 3 + for,,,
regex_Gleason_with_plus2 = "Gleason\s?(?:score)?:?\s*(\d{1})(?:\s|\.)(\d{1})" # Gleason 4.4, Gleason score: 3 3 (without "+", but assuming typo)
regex_Gleason_wo_plus = "(?:Gleason\s?|G|GS)\s?(?:score)?:?\s*(5|6|7|8|9|10)\s?(?!\+)" # Gleason score 7, Gleason: 10, G7
regex_Gleason_grade = "(?:Gleason grade|G\s?G|grade group|group grade)(?:\sscore\s?)?:?\s?(\d{1})" # Gleason grade

def extract_PreMRI_worstGleason(text):
    """
    main function
    :param text: Hx section text
    :return: highest Gleason score. if >=11, manual check.
    """
    if pd.isnull(text):
        return np.NaN
    else:
        # try to extract Gleason score
        a = extract_worst_Gleason_convert(text, regex=regex_Gleason_with_plus)
        b = extract_worst_Gleason_convert(text, regex=regex_Gleason_with_plus2)
        c = extract_worst_Gleason(text, regex = regex_Gleason_wo_plus)
        max_ = pd.Series([a,b,c]).fillna(0).max()
        if max_ == 0:
            # In cases without Gleason score, try to extract Gleason grade (1-5) and Gleason describing keywords.
            max_grade = extract_worst_Gleason(text, regex=regex_Gleason_grade)
            grade_score_convert_dic = {1:6,2:7,3:7,4:8,5:9} # grade 1-> score6,,,,,grade 5 -> score 9
            try:
                max_ = grade_score_convert_dic[max_grade]
            except:
                max_ = np.NaN
            if re.findall("unfavorable(?:-|\s)intermediate", text, re.IGNORECASE):
                word_max = 8
            elif re.findall("^(un)favorable(?:-|\s)intermediate", text, re.IGNORECASE):
                word_max = 7
            else:
                word_max = np.NaN
            max_ = pd.Series([max_,word_max]).fillna(0).max()
            if max_ == 0:
                max_ = np.NaN

        if max_ >= 11:
            return "manual_check"
        else:
            return max_

def convert_Gleason(regex_Gleason_results):
    """
    subfunction 1
    :param regex_Gleason_results:
    :return:
    """
    try:
        a = regex_Gleason_results[0]
        b = regex_Gleason_results[1]
        if a == "for":
            a = 4
        if b=="for":
            b = 4
        #text = f"{a + b} ({a}+{b})"
        return int(a) + int(b)
    except:
        return np.NaN

def extract_worst_Gleason(text, regex):
    """
    subfunction 2
    :param regex_Gleason_results:
    :return:
    """
    if pd.isnull(text):
        return np.NaN
    else:
        result_list =re.findall(regex, text, re.IGNORECASE)
        if len(result_list)==0:
            return np.NaN
        else:
            return max([int(x) for x in result_list])

def extract_worst_Gleason_convert(text, regex):
    """
    subfunction 3
    :param regex_Gleason_results:
    :return:
    """
    if pd.isnull(text):
        return np.NaN
    else:
        result_list =re.findall(regex, text, re.IGNORECASE)
        if len(result_list)==0:
            return np.NaN
        else:
            return max([convert_Gleason(x) for x in result_list])

############ ADC values ############
def extract_ADC(DWI_text):
    """
    extract the first-appeared 3 or 4-lengths digits after "ADC"
    :param DWI_text:
    :return:
    """
    try:
        ADC_index = re.search("ADC", DWI_text).span()[0]
        ADC_value = re.search("\d{3,4}", DWI_text[ADC_index:])[0]
        return int(ADC_value)
    except:
        return np.NaN

############ Magnetic field strength ############
def extract_magnetic_strength(MRI_technique_text):
    magnet_regex = "(3|3.0|1.5)\s?(?:Tesla|T|tesla)"
    try:
        return float(re.findall(magnet_regex, MRI_technique_text)[0])
    except:
        return np.NaN

#%%
############ convert the degree of confidence of LN-metastasis and local staging #################
def search_keywords_and_convert_into_score(results, local_staging_col, local_staging_LN_dic):
    local_staging_keywords = "|".join(list(local_staging_LN_dic.keys()))
    results[f"{local_staging_col}_kw"] = results[local_staging_col].fillna("").apply(lambda text:[x.lower() for x in list(set(re.findall(local_staging_keywords, text, re.IGNORECASE)))])
    # If contain "intact" or "abasent" -> intact
    intact_exist = results[f"{local_staging_col}_kw"].apply(lambda kw_list: ("negative" in [x.lower() for x in kw_list]) or
                                                                            ("intact" in [x.lower() for x in kw_list]) or
                                                                            ("absent" in [x.lower() for x in kw_list]) or
                                                                            ("not definite" in [x.lower() for x in kw_list]) or
                                                                            ("no definite" in [x.lower() for x in kw_list]) or
                                                                            ("no suspicio" in [x.lower() for x in kw_list]) or
                                                                            ("no for suspicio" in [x.lower() for x in kw_list]))
    results.loc[intact_exist, f"{local_staging_col}_kw_converted"] = 0

    # If contain "indeterminate" -> indeterminate
    indeterminate_exist = results[f"{local_staging_col}_kw"].apply(lambda kw_list: ("indeterminate" in [x.lower() for x in kw_list]) or
                                                                                   ("indeterminant" in [x.lower() for x in kw_list]) or
                                                                                   ("low suspicio" in [x.lower() for x in kw_list]))
    results.loc[indeterminate_exist, f"{local_staging_col}_kw_converted"] = 1

    # If multiple keyword extracted, and "extracapsular extension" exists -> remove "extracapsular extension"
    with_ext_extension_and_other_kw = (results[f"{local_staging_col}_kw"].apply(len)>=2) & (results[f"{local_staging_col}_kw"].apply(lambda kw_list: "extracapsular extension" in [x.lower() for x in kw_list]))
    results.loc[with_ext_extension_and_other_kw, f"{local_staging_col}_kw"] = results.loc[with_ext_extension_and_other_kw, f"{local_staging_col}_kw"].apply(lambda kw_list: [x for x in kw_list if x != "extracapsular extension"])

    # convert keyword list into max value using local_staging_dic
    with_kw_wo_intact_indeterminate_index = results[(results[f"{local_staging_col}_kw_converted"].isnull()) & (results[f"{local_staging_col}_kw"].apply(len)>=1)].index
    results.loc[with_kw_wo_intact_indeterminate_index, f"{local_staging_col}_kw_converted"] = results.loc[with_kw_wo_intact_indeterminate_index, f"{local_staging_col}_kw"].apply(lambda kw_list:np.max([local_staging_LN_dic[x.lower()] if x in local_staging_LN_dic.keys() else 0 for x in kw_list])) # no -> 0
    return results

############### LN ############################

direction = ["right", "left", "bilateral"]
location = ["para", "aorto", "obturator", "common", "internal", "external", "iliac", "proxi", "distal", "peri", "pre", "pelvic", "sidewall"]
LN = ["lymph", "node", "LN"]
ln_words = "|".join(direction+location+LN)

### Create location regex ###
# wo laterality
ln_wo_lateral_regex = ["(?:para|peri)?-?aortic", "mesorectal", "peri-?vesical", "retro-?peritoneal", "pre-?sacral", "pre-?vesical", "retro-?pubic"]
ln_wo_lateral_label = ["paraaortic", "mesorectal", "perivesical", "retroperitoneal", "presacral", "prevesical", "retropubic"]
ln_wo_lateral_dic = dict(zip(ln_wo_lateral_regex, ln_wo_lateral_label))
ln_wo_latelal_regex_label_dict = dict([(f"{reg}", f"{loc}") for reg, loc in ln_wo_lateral_dic.items()])

# With laterality
ln_with_lateral_regex = ["(?<!common )(?<!external )(?<!internal )iliac", "common iliac", "external iliac", "internal iliac", "inguinal", "obturator", "pelvic", "sidewall", "peri-?rectal"]
ln_with_lateral_label = ["common iliac", "common iliac", "external iliac", "internal iliac", "inguinal", "obturator", "pelvic", "sidewall", "perirectal"]
ln_with_lateral_dic = dict(zip(ln_with_lateral_regex, ln_with_lateral_label))
ln_with_latelal_regex_label_dict = dict([(f"{d} {reg}", f"{d} {loc}") for d in direction for reg, loc in ln_with_lateral_dic.items()])

# combine the above two dictionary and rename.
ln_wo_latelal_regex_label_dict.update(ln_with_latelal_regex_label_dict)
ln_regex_label_dict = ln_wo_latelal_regex_label_dict

def extract_LN_text(text, ln_words = ln_words):
    ln_words_list = list(re.finditer(ln_words, text, re.IGNORECASE))
    try:
        text_beg = min(x.span()[0] for x in ln_words_list)
        text_end = max(x.span()[1] for x in ln_words_list)
        return text[text_beg:text_end]
    except:
        return ""


def extract_LN_location(LN_text, ln_regex_label_dict=ln_regex_label_dict):
    """
    from LN_text, extract LN location label.
    :param LN_text: after "extract_LN_text" function
    :param ln_regex_label_dict: store the regex and corresponding location label
    :return:
    """
    return [label for regex, label in ln_regex_label_dict.items() if re.search(regex, LN_text, re.IGNORECASE)]

def extract_LN_loc_cm(list_of_text):
    """
    from splited text of the LN setction, create list of tuple having location and length (cm).
    use "extract_LN_text" and "extract_LN_location" as subfunctions.
    :param list_of_text:
    :return: dictionary of location and length
    """
    list_of_tuple_loc_cm = [(list(set(extract_LN_location(extract_LN_text(text)))), max_if_exist(extract_cm(text))) for text in list_of_text if len(extract_LN_location(extract_LN_text(text)))>=1]

    # separate multi location
    lis = []
    for tup in list_of_tuple_loc_cm:
        location_list = tup[0]
        max_cm = tup[1]
        # if length was not extracted -> only mention the existence
        if pd.isnull(max_cm):
            max_cm = "mentioned_but_length_not_extracted"
        for location in location_list:
            lis.append((location, max_cm))
    location_length_dict = dict(lis)

    # convert bilateral to right and left
    location_length_dict_wo_bilateral = {}
    for k, v in location_length_dict.items():
        if "bilateral" in k:
            for direction in ["right", "left"]:
                location_length_dict_wo_bilateral[k.replace("bilateral", direction)] = v
        else:
            location_length_dict_wo_bilateral[k] = v
    return location_length_dict_wo_bilateral

def extract_LN_main(LN_section_text):
    if pd.isnull(LN_section_text):
        return np.NaN
    else:
        text_list = re.split("\. |, | and |;|\) ", LN_section_text)
        location_length_dict = extract_LN_loc_cm(text_list)
        return location_length_dict

def is_post_treatment(report):
    # This function was created by Dr. Kurata to fix the post-treatment status in March 2025. 
    # ポジティブなキーワードと、そのキーワードに対応する除外パターンのマッピング
    positive_exclusion_map = {
        # recurrence などの場合、その周辺に感染症関連（例：urinary tract infections など）があると偽陽性
        r"(?i)(recurrence|recurrent|relapse)": [
            r"(?i)urosepsis",
            r"(?i)infections?",
            r"(?i)urinary tract (?:infections?|symptoms)",
            r"(?i)prostatitis",
            r"(?i)UTIs?",
            r"(?i)cystitis",
            r"(?i)hernias?"
        ],
        r"(?i)(Prostatectomy|Radiation|HIFU|ADT|blation|proton|sbrt|imrt|rtx|radiotherapy|brachy|anti-androgen|lupron|androgen deprivation|adt|hormone|hormonal|brachytherapy|high-intensity focused ultrasound|TULSA|brachial therapy|radiated|nano knife|radiotherapy|beam|rrp|rarp)": [
            r"(?i)contemplat",
            r"(?i)evaluat",
            r"(?i)plan",
            r"(?i)schedul",
            r"(?i)attempt",
            r"(?i)staging prior to",
            r"(?i)discuss",
            r"(?i)Assessment",
            r"(?i)pending",
            r"(?i)prepar",
            r"(?i)oncology service",
            r"(?i)consider",
            r"(?i)review"
        ],
        r"(?i)((post|following) prostate cancer treatment|nadir|s/p RT|vesicourethral|vesico-urethral|treatment planning|therapy planning|planning purposes|fiducial marker|carbon marker|fiducial markers|carbon markers|targeting marker|treatment position|dosimetry|hydrogel|choline)":[
 
        ]
    
    }
    
    sentences = re.split(r'(?<=[.!?])\s+', report)
    
    for sentence in sentences:
        for positive_regex, exclusion_list in positive_exclusion_map.items():
            if re.search(positive_regex, sentence):
                # 該当するポジティブキーワードが見つかった場合、対応する除外パターンが同じ文章に存在するかチェック
                exclude = False
                for excl in exclusion_list:
                    if re.search(excl, sentence):
                        exclude = True
                        break
                # 除外パターンがなければ、この文章は有効な陽性結果
                if not exclude:
                    return True
    return False
