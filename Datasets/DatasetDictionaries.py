import numpy as np

BinningDict = {
#    "mt": np.linspace(0.0, 1000.0, 51),
#    "mt": np.array(np.linspace(0.0, 400.0, 21).tolist() + [500.0, 1000.0, 3000.0]),
    "mt": np.logspace(0.0, 3.3, 10),
    "MET": np.linspace(0.0, 1000.0, 51),
}

InvertedEventTypeDict = {
    0: "Signal",
    1: "TT",
    2: "DY",
    3: "QCD",
    4: "ST",
    5: "WJets",
    6: "Diboson"
}

EventTypeDict = {
    "ChargedHiggs_": 0,
#    "QCD_HT50to100": 3,
#    "QCD_HT100to200": 3,
    "QCD_HT200to300": 3,
    "QCD_HT300to500": 3,
    "QCD_HT500to700": 3,
    "QCD_HT700to1000": 3,
    "QCD_HT1000to1500": 3,
    "QCD_HT1500to2000": 3,
    "QCD_HT2000toInf": 3,
    "ST_s_channel_4f": 4,
    "ST_t_channel_antitop_4f": 4,
    "ST_t_channel_top_4f": 4,
    "ST_tW_antitop_5f": 4,
    "ST_tW_top_5f": 4,
    "WJetsToLNu": 5,
#    "WJetsToLNu_HT_0To70": 5,
    "WJetsToLNu_HT_70To100": 5,
    "WJetsToLNu_HT_100To200": 5,
    "WJetsToLNu_HT_200To400": 5,
    "WJetsToLNu_HT_400To600": 5,
    "WJetsToLNu_HT_600To800": 5,
    "WJetsToLNu_HT_800To1200": 5,
    "WJetsToLNu_HT_1200To2500": 5,
    "WJetsToLNu_HT_2500ToInf": 5,
    "TT": 1,
    "ZZ": 6,
    "WZ": 6,
    "WW": 6,
    "WWToLNuQQ": 6,
    "WWTo2L2Nu": 6,
    "WWTo4Q": 6,
    "DYJetsToLL_M_50": 2
}

XSecDict = {
    "QCD_HT50to100": 2.464e+08,
    "QCD_HT100to200": 2.803e+07,
    "QCD_HT200to300": 1.713e+06,
    "QCD_HT300to500": 3.475e+05,
    "QCD_HT500to700": 3.208e+04,
    "QCD_HT700to1000": 6.833e+03,
    "QCD_HT1000to1500": 1.208e+03,
    "QCD_HT1500to2000": 1.201e+02,
    "QCD_HT2000toInf": 2.526e+01,
    "ST_s_channel_4f": 11.36,
    "ST_t_channel_antitop_4f": 80.95,
    "ST_t_channel_top_4f": 136.02,
    "ST_tW_antitop_5f": 35.85,
    "ST_tW_top_5f": 35.85,
    "WJetsToLNu": 20508.9 * 3,
    "WJetsToLNu_HT_0To70": 20508.9 * 3,
    "WJetsToLNu_HT_70To100": 1.353e+03 * 1.2138,
    "WJetsToLNu_HT_100To200": 1.293e+03 * 1.2138,
    "WJetsToLNu_HT_200To400": 3.86e+02 * 1.2138,
    "WJetsToLNu_HT_400To600": 47.9 * 1.2138,
    "WJetsToLNu_HT_600To800": 12.8 * 1.2138,
    "WJetsToLNu_HT_800To1200": 5.26 * 1.2138,
    "WJetsToLNu_HT_1200To2500": 1.33 * 1.2138,
    "WJetsToLNu_HT_2500ToInf": 3.089e-02 * 1.2138,
    "TT": 831.76,
    "ZZ": 16.523,
    "WZ": 47.13,
    "WW": 118.7,
    "WWToLNuQQ": 49.997,
    "WWTo2L2Nu": 12.178,
    "WWTo4Q": 51.723,
    "DYJetsToLL_M_50": 1921.8 * 3.0
}