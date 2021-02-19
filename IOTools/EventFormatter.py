import uproot
import pandas as pd
import numpy as np
class EventFormatter():

    def __init__(self):
        self.used_variables = ["event",
                        "nGoodOfflineVertices",
                        "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET90_vx",
                        "MET_Type1_x", "MET_Type1_y",
                        "L1MET_pat_x", "L1MET_pat_y",
                        "Taus_pt", "Taus_eta", "Taus_phi", "Taus_e",
                        "Taus_decayModeFindingNewDMs", "Taus_againstElectronTightMVA6",
                        "Taus_againstMuonLoose3", "Taus_byLooseIsolationMVArun2v1DBoldDMwLT",
                        "Taus_lChTrkPt", "Taus_lChTrkEta", "Taus_nProngs", "Taus_pdgId",
                        "Electrons_pt", "Electrons_eta", "Electrons_phi", "Electrons_e", "Electrons_MVA", "Electrons_effAreaMiniIso",
                        "Muons_pt", "Muons_eta", "Muons_phi", "Muons_e", "Muons_muIDLoose", "Muons_effAreaMiniIso",
                        "Jets_pt", "Jets_eta", "Jets_phi", "Jets_e", "Jets_IDloose", "Jets_pfCombinedInclusiveSecondaryVertexV2BJetTags",

#                        "HLTBJet_pt", "HLTBJet_eta", "HLTBJet_phi", "HLTBJet_e",

                        "METFilter_hbheNoiseTokenRun2Loose",
                        "METFilter_Flag_HBHENoiseFilter",
                        "METFilter_Flag_HBHENoiseIsoFilter",
                        "METFilter_Flag_EcalDeadCellTriggerPrimitiveFilter",
                        "METFilter_Flag_eeBadScFilter",
                        "METFilter_Flag_goodVertices",
                        "METFilter_Flag_globalTightHalo2016Filter",
                        "METFilter_badPFMuonFilter",
                        "METFilter_badChargedCandidateFilter"]

    def format_file(self, filepath):
        '''
        Reads a single file and formats it into pandas dataframe
        :param filepath: path to file
        :return: pandas dataframe
        '''
        print(filepath)
        with uproot.open(filepath+":Events", workers=12) as tree:
            print(tree.keys())
            df = pd.DataFrame(tree.arrays(self.used_variables, library="np"), columns=self.used_variables)
            num_jets = [len(x) for x in df["Jets_eta"]]
            num_taus = [len(x) for x in df["Taus_eta"]]
            num_electrons = [len(x) for x in df["Electrons_eta"]]
            num_muons = [len(x) for x in df["Muons_eta"]]
            df["met"] = np.sqrt(np.square(df["MET_Type1_x"].values)+np.square(df["MET_Type1_y"].values))
            df["l1met"] = np.sqrt(np.square(df["L1MET_pat_x"].values)+np.square(df["L1MET_pat_y"].values))
            df["met_phi"] = np.arctan2(df["MET_Type1_x"].values, df["MET_Type1_y"].values)
            df["met_passed_filters"] =
