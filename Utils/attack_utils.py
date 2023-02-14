import numpy as np
import pandas as pd

from Attacks.HopSkipJump_tabilar import HopSkipJump


class FactoryCat:
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def method(self, vector):
        vector = integer(vector)
        return bound(vector, self.min, self.max)

    def get_method(self):
        return self.method


def bound(vector, v_min, v_max):
    v = np.clip(vector, v_min, v_max)
    return v


def normalized(vector, v_min=0, v_max=0):
    return bound(vector, 0.0, 1.0)


def positive(vector, v_min=0, v_max=0):
    return bound(vector, 0, np.inf)


def integer(vector, v_min=0, v_max=0):
    return np.round(vector)


def binary(vector, v_min=0, v_max=0):
    vector = integer(vector)
    return bound(vector, 0, 1)


def negative(vector, v_min=0, v_max=0):
    return bound(vector, -np.inf, 0)


def categorical(vector, min_v, max_v):
    return FactoryCat(min_v, max_v).get_method()


def cond(vector, cond=None, val=0):
    if (cond != None):
        if(eval(str(vector)+str(cond))):
            v = np.clip(vector, val, val)
        return v

def bmi(vector, height, weight):
    if (weight > 0):
        return height/weight

def increas(vector, pre_vector):
    return np.greater(vector, pre_vector)

def get_constrains(dataset_name, perturbability_path):
    perturbability = pd.read_csv(perturbability_path)
    perturbability = perturbability["Perturbability"].to_numpy()

    if dataset_name == "HATE":
        constrains = {
            'hate_neigh': [binary],
            'anger_empath': [normalized], ###
            'appearance_empath': [normalized], ###
            'attractive_empath': [normalized], ###
            'banking_empath': [positive], ###
            'betweenness': [positive], ###
            'body_empath': [normalized], ###
            'c_air_travel_empath': [normalized], ###
            'c_art_empath': [normalized], ###
            'c_banking_empath': [normalized], ###
            'c_betweenness': [positive], ###
            'c_childish_empath': [normalized], ###
            'c_cleaning_empath': [normalized],
            'c_computer_empath': [normalized],
            'c_crime_empath': [normalized],
            'c_dispute_empath': [normalized], ###
            'c_divine_empath': [normalized],
            'c_dominant_personality_empath': [normalized], ###
            'c_eigenvector': [positive], ###
            'c_exasperation_empath': [normalized],
            'c_family_empath': [normalized], ###
            'c_farming_empath': [normalized],
            'c_fashion_empath': [normalized], ###
            'c_fire_empath': [normalized],
            'c_followees_count': [positive], ###
            'c_followers_count': [positive],
            'c_furniture_empath': [normalized],
            'c_hate_empath': [normalized],
            'c_hipster_empath': [normalized],
            'c_hygiene_empath': [normalized],
            'c_independence_empath': [normalized],
            'c_irritability_empath': [normalized],
            'c_joy_empath': [normalized],
            'c_kill_empath': [normalized],
            'c_legend_empath': [normalized],
            'c_listen_empath': [normalized],
            'c_lust_empath': [normalized],
            'c_masculine_empath': [normalized],
            'c_medical_emergency_empath': [normalized],
            'c_medieval_empath': [normalized],
            'c_money_empath': [normalized],
            'c_number urls': [positive],
            'c_optimism_empath': [normalized],
            'c_out_degree': [normalized],
            'c_power_empath': [normalized],
            'c_rage_empath': [normalized],
            'c_ridicule_empath': [normalized],
            'c_shape_and_size_empath': [normalized],
            'c_ship_empath': [normalized],
            'c_sleep_empath': [normalized],
            'c_social_media_empath': [normalized],
            'c_superhero_empath': [normalized],
            'c_swearing_terms_empath': [normalized],
            'c_time_diff': [positive],
            'c_tourism_empath': [normalized],
            'c_traveling_empath': [normalized],
            'c_trust_empath': [normalized],
            'c_weakness_empath': [normalized],
            'c_wedding_empath': [normalized],
            'c_work_empath': [normalized],
            'car_empath': [positive],
            'cleaning_empath': [normalized],
            'competing_empath': [normalized],
            'created_at': [positive, integer],
            'disgust_empath': [normalized],
            'economics_empath': [normalized],
            'fabric_empath': [normalized],
            'favorites_count': [positive, integer],
            'friends_empath': [normalized],
            'gain_empath': [normalized],
            'health_empath': [normalized],
            'help_empath': [normalized],
            'home_empath': [normalized],
            'horror_empath': [normalized],
            'irritability_empath': [normalized],
            'joy_empath': [normalized],
            'law_empath': [normalized],
            'leader_empath': [normalized],
            'liquid_empath': [normalized],
            'lust_empath': [normalized],
            'magic_empath': [normalized],
            'morning_empath': [normalized],
            'music_empath': [normalized],
            'musical_empath': [normalized],
            'nervousness_empath': [normalized],
            'normal_neigh': [binary],
            'real_estate_empath': [normalized],
            'sentiment': [positive],
            'sexual_empath': [normalized],
            'sleep_empath': [normalized],
            'sound_empath': [normalized],
            'sports_empath': [normalized],
            'statuses_count': [positive, integer],
            'surprise_empath': [normalized],
            'tourism_empath': [normalized],
            'urban_empath': [normalized],
            'vacation_empath': [normalized],
            'warmth_empath': [normalized],
            'work_empath': [normalized],
            'youth_empath': [normalized],
            'zest_empath': [normalized]
        }

    elif dataset_name == "CREDIT":
        constrains = {
            'AMT_ANNUITY': [positive],
            'AMT_CREDIT': [positive],
            'AMT_CREDIT_MAX_OVERDUE': [positive],
            'AMT_GOODS_PRICE': [positive],
            'AMT_INCOME_TOTAL': [positive],
            'AMT_REQ_CREDIT_BUREAU_HOUR': [positive, integer],
            'AMT_REQ_CREDIT_BUREAU_QRT': [positive, integer],
            'AMT_REQ_CREDIT_BUREAU_WEEK': [positive, integer],
            'APARTMENTS_AVG': [normalized],
            'APARTMENTS_MEDI': [normalized],
            'APARTMENTS_MODE': [normalized],
            'CODE_GENDER': [binary],
            'DAYS_BIRTH': [integer, negative],
            'DAYS_EMPLOYED': [integer, negative],
            'DAYS_ID_PUBLISH': [integer, negative],
            'DAYS_LAST_PHONE_CHANGE': [integer, negative],
            'DAYS_REGISTRATION': [integer, negative],
            'DEF_30_CNT_SOCIAL_CIRCLE': [integer, positive],
            'DEF_60_CNT_SOCIAL_CIRCLE': [integer, positive],
            'ENTRANCES_AVG': [normalized],
            'ENTRANCES_MEDI': [normalized],
            'EXT_SOURCE_1': [normalized],
            'EXT_SOURCE_2': [normalized],
            'EXT_SOURCE_3': [normalized],
            'FLAG_DOCUMENT_3': [binary],
            'FLAG_DOCUMENT_8': [binary],
            'FLAG_DOCUMENT_18': [binary],
            'FLAG_OWN_CAR': [binary],
            'FLAG_WORK_PHONE': [binary],
            'FLOORSMAX_AVG': [normalized],
            'FLOORSMAX_MEDI': [normalized],
            'FLOORSMAX_MODE': [normalized],
            'LIVE_CITY_NOT_WORK_CITY': [binary],
            'LIVE_REGION_NOT_WORK_REGION': [binary],
            'LIVINGAREA_AVG': [normalized],
            'NAME_CONTRACT_TYPE': [binary],
            'NAME_EDUCATION_TYPE': [categorical(0, 4)],
            'NAME_FAMILY_STATUS': [categorical(0, 4)],
            'NAME_HOUSING_TYPE': [categorical(0, 5)],
            'NAME_INCOME_TYPE': [categorical(0, 7)],
            'NAME_TYPE_SUITE': [categorical(-1, 6)],
            'OCCUPATION_TYPE': [categorical(-1, 17)],
            'ORGANIZATION_TYPE': [categorical(0, 57)],
            'OWN_CAR_AGE': [integer, positive],
            'REG_CITY_NOT_LIVE_CITY': [binary],
            'REGION_POPULATION_RELATIVE': [normalized],
            'REGION_RATING_CLIENT': [categorical(1, 3)],
            'REGION_RATING_CLIENT_W_CITY': [categorical(1, 3)],
            'TOTALAREA_MODE': [normalized],
            'WALLSMATERIAL_MODE': [categorical(-1, 6)],
            'WEEKDAY_APPR_PROCESS_START': [categorical(0, 6)],
            'YEARS_BEGINEXPLUATATION_MEDI': [normalized]
        }

    # "ICU" dataset
    elif dataset_name == "ICU":
        constrains = {
            'hospital_id':[positive(v_min=1),integer],	
            'age':[positive,integer],
            'bmi':[bmi],
            'elective_surgery':[binary],
            'ethnicity':[integer,categorical(0,6)],
            'height':[positive],
            'hospital_admit_source':[integer,categorical(0,14)],
            'icu_admit_source':[integer,categorical(0,4)],
            'icu_id':[integer,positive],
            'icu_stay_type':[integer,categorical(0,2)],
            'icu_type':[integer,categorical(0,7)],
            'weight':[positive, increas],
            'albumin_apache':[positive],
            'apache_2_diagnosis':[positive(v_min=100)],
            'apache_3j_diagnosis':[positive],
            'arf_apache':[binary],
            'bun_apache':[positive],
            'creatinine_apache':[positive],
            'fio2_apache':[normalized(0,1)],
            'gcs_eyes_apache':[integer,categorical(1,4)],
            'gcs_motor_apache':[integer,categorical(1,6)],
            'gcs_unable_apache':[binary],
            'gcs_verbal_apache':[integer,categorical(1,5)],
            'glucose_apache':[positive,integer],
            'heart_rate_apache':[positive,integer(v_min=1)],
            'hematocrit_apache':[positive],
            'intubated_apache':[binary],
            'map_apache':[positive,integer(v_min=1)],
            'resprate_apache':[positive],
            'sodium_apache':[positive],
            'temp_apache':[positive],
            'urineoutput_apache':[positive],
            'ventilated_apache':[binary],
            'wbc_apache':[positive],
            'd1_diasbp_invasive_max':[positive,integer],
            'd1_diasbp_invasive_min':[positive,integer],
            'd1_heartrate_max':[positive,integer],
            'd1_heartrate_min':[positive,integer],
            'd1_mbp_max':[positive,integer],
            'd1_mbp_min':[positive,integer],
            'd1_resprate_max':[positive,integer],
            'd1_spo2_min':[positive,integer],
            'd1_sysbp_max':[positive,integer],
            'd1_sysbp_min':[positive,integer],
            'd1_temp_max':[positive],
            'd1_temp_min':[positive],
            'h1_diasbp_max':[positive,integer],
            'h1_diasbp_min':[positive,integer],
            'h1_heartrate_max':[positive,integer],
            'h1_heartrate_min':[positive,integer],
            'h1_mbp_max':[positive,integer],
            'h1_mbp_min':[positive,integer],
            'h1_resprate_max':[positive,integer],
            'h1_sysbp_max':[positive,integer],
            'h1_sysbp_min':[positive,integer],
            'h1_temp_max':[positive],
            'h1_temp_min':[positive],
            'd1_albumin_max':[positive],
            'd1_calcium_max':[positive],
            'd1_calcium_min':[positive],
            'd1_glucose_min':[positive(v_min=1),integer],
            'd1_hco3_max':[positive,integer],
            'd1_hco3_min':[positive,integer],
            'd1_hemaglobin_max':[positive],
            'd1_hemaglobin_min':[positive],
            'd1_platelets_max':[positive(v_min=1),integer],
            'd1_potassium_max':[positive],
            'd1_potassium_min':[positive],
            'd1_sodium_max':[positive,integer],
            'd1_sodium_min':[positive,integer],	
            'h1_glucose_max':[positive,integer],
            'd1_arterial_pco2_max':[positive],
            'd1_arterial_ph_min':[positive],
            'd1_arterial_po2_min':[positive],
            'd1_pao2fio2ratio_min':[positive],
            'apache_4a_hospital_death_prob':[normalized(-1,1),cond('<0',-1)],
            'apache_4a_icu_death_prob':[normalized(-1,1),cond('<0',-1)],
            'cirrhosis'	:[binary],
            'diabetes_mellitus'	:[binary],
            'hepatic_failure':[binary],
            'immunosuppression':[binary],
            'leukemia':[binary],
            'solid_tumor_with_metastasis':[binary],
            'apache_3j_bodysystem':[integer,categorical(0,10)],
            'apache_2_bodysystem':[integer,categorical(0,8)],
        }
            
    
    else:
        constrains = {
            'age': [integer, positive],
            'bmi': [positive],
            'elective_surgery': [categorical(1, 2)],
            'height': [positive],
            'hospital_admit_source': [categorical(1, 15)],
            'icu_admit_source': [categorical(1, 5)],
            'icu_stay_type': [categorical(1, 3)],
            'icu_type': [categorical(1, 8)],
            'weight': [positive],
            'albumin_apache': [positive],
            'apache_2_diagnosis': [categorical(1, 44)],
            'apache_3j_diagnosis': [positive],
            'apache_post_operative': [categorical(1, 2)],
            'arf_apache': [categorical(1, 2)],
            'bilirubin_apache': [positive],
            'bun_apache': [positive],
            'creatinine_apache': [positive],
            'fio2_apache': [positive, normalized],
            'gcs_eyes_apache': [integer, positive],
            'gcs_motor_apache': [integer, positive],
            'gcs_unable_apache': [categorical(1, 2)],
            'gcs_verbal_apache': [integer, positive],
            'glucose_apache': [positive],
            'heart_rate_apache': [integer, positive],
            'hematocrit_apache': [positive],
            'intubated_apache': [categorical(1, 2)],
            'map_apache': [integer, positive],
            'paco2_apache': [positive],
            'pao2_apache': [positive],
            'ph_apache': [positive],
            'resprate_apache': [positive],
            'sodium_apache': [positive],
            'temp_apache': [positive],
            'urineoutput_apache': [positive],
            'ventilated_apache': [categorical(1, 2)],
            'wbc_apache': [positive],
            'd1_diasbp_invasive_max': [integer, positive],
            'd1_diasbp_invasive_min': [integer, positive],
            'd1_diasbp_max': [integer, positive],
            'd1_diasbp_min': [integer, positive],
            'd1_heartrate_max': [integer, positive],
            'd1_heartrate_min': [integer, positive],
            'd1_mbp_invasive_max': [integer, positive],
            'd1_mbp_invasive_min': [integer, positive],
            'd1_mbp_max': [integer, positive],
            'd1_mbp_min': [integer, positive],
            'd1_resprate_max': [integer, positive],
            'd1_resprate_min': [integer, positive],
            'd1_spo2_max': [integer, positive],
            'd1_spo2_min': [integer, positive],
            'd1_sysbp_invasive_max': [integer, positive],
            'd1_sysbp_invasive_min': [integer, positive],
            'd1_sysbp_max': [integer, positive],
            'd1_sysbp_min': [integer, positive],
            'd1_temp_max': [positive],
            'd1_temp_min': [positive],
            'h1_diasbp_max': [integer, positive],
            'h1_diasbp_min': [integer, positive],
            'h1_diasbp_noninvasive_min': [integer, positive],
            'h1_heartrate_max': [integer, positive],
            'h1_heartrate_min': [integer, positive],
            'h1_mbp_max': [integer, positive],
            'h1_mbp_min': [integer, positive],
            'h1_resprate_max': [integer, positive],
            'h1_resprate_min': [integer, positive],
            'h1_spo2_max': [integer, positive],
            'h1_spo2_min': [integer, positive],
            'h1_sysbp_max': [integer, positive],
            'h1_sysbp_min': [integer, positive],
            'h1_sysbp_noninvasive_min': [integer, positive],
            'h1_temp_max': [positive],
            'h1_temp_min': [positive],
            'd1_albumin_max': [positive],
            'd1_albumin_min': [positive],
            'd1_bilirubin_min': [positive],
            'd1_bun_min': [positive],
            'd1_calcium_max': [positive],
            'd1_calcium_min': [positive],
            'd1_creatinine_min': [positive],
            'd1_glucose_max': [integer, positive],
            'd1_glucose_min': [integer, positive],
            'd1_hco3_max': [positive],
            'd1_hco3_min': [positive],
            'd1_hemaglobin_max': [positive],
            'd1_hemaglobin_min': [positive],
            'd1_hematocrit_max': [positive],
            'd1_hematocrit_min': [positive],
            'd1_inr_max': [positive],
            'd1_inr_min': [positive],
            'd1_lactate_max': [positive],
            'd1_lactate_min': [positive],
            'd1_platelets_max': [integer, positive],
            'd1_platelets_min': [positive],
            'd1_potassium_max': [positive],
            'd1_potassium_min': [positive],
            'd1_sodium_max': [positive],
            'd1_sodium_min': [positive],
            'd1_wbc_max': [positive],
            'd1_wbc_min': [positive],
            'h1_glucose_max': [positive],
            'h1_glucose_min': [integer, positive],
            'h1_hemaglobin_max': [positive],
            'h1_hemaglobin_min': [positive],
            'h1_potassium_max': [positive],
            'h1_potassium_min': [positive],
            'h1_sodium_max': [positive],
            'h1_sodium_min': [positive],
            'd1_arterial_pco2_max': [positive],
            'd1_arterial_pco2_min': [positive],
            'd1_arterial_ph_max': [positive],
            'd1_arterial_ph_min': [positive],
            'd1_arterial_po2_max': [positive],
            'd1_arterial_po2_min': [positive],
            'd1_pao2fio2ratio_max': [positive],
            'd1_pao2fio2ratio_min': [positive],
            'aids': [categorical(1, 2)],
            'cirrhosis': [categorical(1, 2)],
            'diabetes_mellitus': [categorical(1, 2)],
            'hepatic_failure': [categorical(1, 2)],
            'immunosuppression': [categorical(1, 2)],
            'leukemia': [categorical(1, 2)],
            'lymphoma': [categorical(1, 2)],
            'solid_tumor_with_metastasis': [categorical(1, 2)],
            'apache_3j_bodysystem': [categorical(1, 12)],
            'apache_2_bodysystem': [categorical(1, 11)],
        }



    return constrains, perturbability


def get_hopskipjump(classifier, constraints, columns_names):
    hsj = HopSkipJump(classifier=classifier,
                      batch_size=64,
                      targeted=False,
                      norm=2,
                      max_iter=50,
                      max_eval=10000,
                      init_eval=100,
                      init_size=100,
                      verbose=True,
                      types2indices=constraints,
                      columns_names=columns_names)
    return hsj
