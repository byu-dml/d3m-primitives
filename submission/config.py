import os

# This is where the datasets directory lives inside
# the docker container.
DATASETS_DIR = "/datasets/seed_datasets_current"
WORKER_ID = os.getenv('WORKER_ID')

PROBLEM_BLACKLIST = {
    "LL1_ArrowHead",  # time series data
    "LL1_TXT_CLS_3746_newsgroup",  # requires reading a text file from csv
    "uu_101_object_categories",  # requires reading an image file from csv
    "57_hypothyroid",
    "uu10_posts_3",  # missing DataSplits file
    "LL1_OSULeaf",  # requires reading a text file from csv
    "LL1_FaceFour",  # requires reading a text file from csv
    "31_urbansound",
    "LL1_multilearn_emotions",  # weird indexes
    "124_174_cifar10",  # images
    "LL1_FISH",  # images
    "124_214_coil20",  # images
    "LL1_Haptics",  # multiple files
    "LL1_VID_UCF11",  # music files
    "LL1_Cricket_Y",  # multi file
    "LL1_ElectricDevices",  # multi file
    "LL1_3476_HMDB_actio_recognition",
    "124_188_usps",  # image data
    "uu1_datasmash",
    "LL1_50words",  # multi-file
    "30_personae",
    "LL1_crime_chicago",
    "LL1_HandOutlines",  # multi file
    "LL0_186_braziltourism",
    "22_handgeometry",
    "uu2_gp_hyperparameter_estimation" "uu2_gp_hyperparameter_estimation_v2",
    "LL0_1220_click_prediction_small",
    "LL1_336_MS_Geolife_transport_mode_prediction",  # too long
    "1567_poker_hand",  # too long: ERROR: BSON document too large (19826399 bytes)
    "LL0_1569_poker_hand",  # Memory error in NP dot product
    "LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon",  # too long
    # GPS data in another file
    "LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction",
    # Large, takes a long time: ~150k rows.
    "SEMI_1217_click_prediction_small"
    "66_chlorineConcentration",
    "LL0_1485_madelon",  # too long
    "LL0_1468_cnae_9",  # also too long
    "LL0_155_pokerhand",  # too long
    "LL0_300_isolet",  # too long
    "LL0_312_scene",  # too long
    "LL0_1113_kddcup99",  # too long
    "LL0_180_covertype",  # too long
    "LL0_1122_ap_breast_prostate",  # too long
    "LL0_180_covertype",  # too long
    "LL0_4541_diabetes130us",  # calculation too big, memory error np.dot()
    "LL0_1457_amazon_commerce_reviews",  # timeouts
    "LL0_1176_internet_advertisements",  # timeouts
    "LL0_1036_sylva_agnostic",  # timeouts
    "LL0_1041_gina_prior2",  # timeouts
    "LL0_1501_semeion",  # timeouts
    "LL0_1038_gina_agnostic",  # timeouts
    "LL0_23397_comet_mc_sample",  # timeouts
    "LL0_1040_sylva_prior",  # same for the rest
    "LL0_1476_gas_drift",
    "LL0_4541_diabetes130us",
    "LL0_12_mfeat_factors",
    "LL0_1515_micro_mass",
    "LL0_1219_click_prediction_small",
    "LL0_4134_bioresponse",
    "LL0_1481_kr_vs_k",
    "LL0_1046_mozilla4",
    "LL0_1471_eeg_eye_state",
    "uu3_world_development_indicators",
    "LL0_344_mv",
    "LL0_574_house_16h",
    "LL0_296_ailerons",
    "LL0_216_elevators",
    "LL0_201_pol",
    "uu2_gp_hyperparameter_estimation_v2",  # has extra data
    "uu2_gp_hyperparameter_estimation",  # has extra data
    "57_hypothyroid",  # always NAN's out
    "LL0_315_us_crime",  # the next are due to timeouts
    "LL0_688_visualizing_soil",
    "LL0_189_kin8nm",
    "LL0_572_bank8fm",
    "LL0_308_puma32h",
    "32_fma",  # audio data
}
