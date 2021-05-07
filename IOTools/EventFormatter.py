import uproot
import pandas as pd
import numpy as np
import awkward as ak
from Utils import FuncTimer,BlockTimer
import numba
import glob
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from Datasets import XSecDict, EventTypeDict


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

    def format_data(self, datafolderpath):
        for name in EventTypeDict.keys():
            # if 'QCD_HT' not in name:
            #     continue

            datasetpath = datafolderpath+"/"+name
            print(name)
            dataframes = self.format_dataset(datasetpath)
            print(dataframes.shape[0])
            savepath = "/home/joona/Documents/preprocessed_HiggsTrainingSets_TEST/" + name + ".pkl"
            pd.to_pickle(dataframes, savepath)

    def format_dataset(self, datasetpath):
        if "ChargedHiggs_" in datasetpath:
            folders = glob.glob(datasetpath + "*")
            filepaths = []
            for folder in folders:
                filepaths = filepaths + glob.glob(folder + "/results/*.root")
        else:
            filepaths = glob.glob(datasetpath + "*/results/*.root")

        # with ProcessPoolExecutor(max_workers=1) as executor:
        #     dataframes = list(executor.map(self.format_file, filepaths))
        dataframes = []
        for filepath in filepaths:
            dataframes.append(self.format_file(filepath))
        if (len(dataframes)>1):
            concatenated_frame = dataframes[0].append(dataframes[1:])
        else:
            concatenated_frame = dataframes[0]
        return concatenated_frame

    def format_file(self, filepath):
        '''
        Reads a single file and formats it into pandas dataframe. Computes high-level variables like the angles
        between objects so that they are available for use in training
        :param filepath: path to file
        :return: pandas dataframe
        '''

        out_columns = ["event", "MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet",
                       "bjetPt", "deltaPhiBjetMet", "TransverseMass", "true_mass"]
        with uproot.open(filepath+":Events", workers=(cpu_count()-1), array_cache=None) as tree:
            is_signal = "ChargedHiggs_" in filepath
            if is_signal:
                _string = filepath.replace('/', '_')
                ind = _string.split("_").index('M') + 1
                true_mass = float(_string.split("_")[ind])

            df_raw = pd.DataFrame(tree.arrays(self.used_variables, library="np"), columns=self.used_variables)
            df_raw = df_raw.dropna()
            df_raw["met"] = np.sqrt(np.square(df_raw["MET_Type1_x"].values)+np.square(df_raw["MET_Type1_y"].values))
            df_raw["l1met"] = np.sqrt(np.square(df_raw["L1MET_pat_x"].values)+np.square(df_raw["L1MET_pat_y"].values))
            df_raw["met_phi"] = np.arctan2(df_raw["MET_Type1_x"].values, df_raw["MET_Type1_y"].values)
            df_raw["met_passed_filters"] = df_raw["METFilter_hbheNoiseTokenRun2Loose"].values & df_raw["METFilter_Flag_HBHENoiseFilter"].values & df_raw["METFilter_Flag_HBHENoiseIsoFilter"].values & df_raw["METFilter_Flag_EcalDeadCellTriggerPrimitiveFilter"].values & df_raw["METFilter_Flag_eeBadScFilter"].values & df_raw["METFilter_Flag_goodVertices"].values & df_raw["METFilter_Flag_globalTightHalo2016Filter"].values & df_raw["METFilter_badPFMuonFilter"].values & df_raw["METFilter_badChargedCandidateFilter"].values

            indices_of_events_passing = self._pass_standard_selections(df_raw)
            if(np.sum(indices_of_events_passing)==0):
                return pd.DataFrame(None, columns=out_columns)
            passed_tau_selection, selected_taus = self._tau_selection(df_raw[indices_of_events_passing])
            if(df_raw[indices_of_events_passing][passed_tau_selection].shape[0] == 0):
                return pd.DataFrame(None, columns=out_columns)
            passed_jet_selection, selected_jets = self._jet_selection(df_raw[indices_of_events_passing][passed_tau_selection])
            if(df_raw[indices_of_events_passing][passed_tau_selection][passed_jet_selection].shape[0] == 0):
                return pd.DataFrame(None, columns=out_columns)
            event_ = df_raw[indices_of_events_passing][passed_tau_selection][passed_jet_selection].loc[:, ["event", "met", "met_phi"]].reset_index(drop=True)
            selected_jets = selected_jets.reset_index(drop=True)
            selected_taus = selected_taus[passed_jet_selection].reset_index(drop=True)
            df_ = pd.concat([event_, selected_taus, selected_jets], axis=1)

            df_["Tau_delta_phi_met"] = df_.apply(add_delta_phi_met, axis=1, args=(df_.columns.get_loc('Tau_phi'),
                                                                                  df_.columns.get_loc('met_phi')))
            df_["Jets_delta_phi_met"] = df_.apply(add_delta_phi_met, axis=1, args=(df_.columns.get_loc('Jets_phi'),
                                                                                   df_.columns.get_loc('met_phi')))
            df_["Jets_bb"] = df_.apply(self.add_back_to_back, axis=1, args=(df_.columns.get_loc("Jets_delta_phi_met"), df_.columns.get_loc("Tau_delta_phi_met")))
            df_["TransverseMass"] = df_.apply(self.add_transverse_mass, axis=1, args=(df_.columns.get_loc("Tau_e"),
                                                                                      df_.columns.get_loc("Tau_delta_phi_met"),
                                                                                      df_.columns.get_loc("met")))
            df_["Bjet_delta_phi_met"] = df_.apply(add_delta_phi_met, axis=1, args=(df_.columns.get_loc('Bjet_phi'),
                                                                                   df_.columns.get_loc('met_phi')))
            df_["Tau_delta_phi_bjet"] = df_.apply(add_delta_phi_met, axis=1, args=(df_.columns.get_loc('Tau_phi'),
                                                                                   df_.columns.get_loc('Bjet_phi')))

            if is_signal:
                df_['true_mass'] = true_mass
            else:
                df_['true_mass'] = df_['TransverseMass']

            df_ = df_[["event", "met", "Tau_pt", "Tau_rtau", "Tau_delta_phi_met", "Tau_delta_phi_bjet", "Bjet_pt", "Bjet_delta_phi_met", "TransverseMass", "true_mass"]]
            out_columns = ["event", "MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet",
                               "bjetPt", "deltaPhiBjetMet", "TransverseMass", "true_mass"]
            df_.columns = out_columns
            final_result = df_
            return final_result

    def add_transverse_mass(self, row, tau_e_ind, tau_delta_phi_met_ind, met_ind):
        tau_e = row[tau_e_ind]
        tau_delta_phi_met = row[tau_delta_phi_met_ind]
        met = row[met_ind]
        return np.sqrt(2 * tau_e * met) * (1 - np.cos(tau_delta_phi_met))

    def add_back_to_back(self, row, jets_dphi_met_ind, tau_dphi_met_ind):
        jets_dphi_met = row[jets_dphi_met_ind]
        tau_dphi_met = row[tau_dphi_met_ind]
        bb = np.sqrt(np.square(jets_dphi_met*360/(2*np.pi))+np.square(tau_dphi_met*360/(2*np.pi)-180))
        return bb

    def _tau_selection(self, df, pt_cut=80.0, eta_cut=1.4, ldg_trk_pt_cut=30.0):
        pt_cut = 60.0           #80.0
        eta_cut = 2.1           #1.4
        ldg_trk_pt_cut = 15.0       #30.0


        input_shape = ak.Array(df["Taus_pt"]).layout.offsets
        decay_modes = np.concatenate(df["Taus_decayModeFindingNewDMs"].ravel()).ravel()
        e_discr = np.concatenate(df["Taus_againstElectronTightMVA6"].ravel()).ravel()
        mu_discr = np.concatenate(df["Taus_againstMuonLoose3"].ravel()).ravel()
        pts = np.concatenate(df["Taus_pt"].ravel()).ravel()
        etas = np.concatenate(df["Taus_eta"].abs().ravel()).ravel()
        isol = np.concatenate(df["Taus_byLooseIsolationMVArun2v1DBoldDMwLT"].ravel()).ravel()
        ldg_trk_pts = np.concatenate(df["Taus_lChTrkPt"].ravel()).ravel()
        ldg_trk_etas = np.concatenate(df["Taus_lChTrkEta"].ravel()).ravel()
        n_prongs = np.concatenate(df["Taus_nProngs"].ravel()).ravel()

        #rtau
        ldg_trk_ptz = np.multiply(np.sinh(ldg_trk_etas), ldg_trk_pts)
        ldg_trk_p = np.sqrt(np.square(ldg_trk_ptz) + np.square(ldg_trk_pts))
        p_tau = pts*np.cosh(etas)

        rtaus = ldg_trk_p/p_tau

        passing = decay_modes
        passing = passing & e_discr
        passing = passing & mu_discr
        passing = passing & (pts > pt_cut)
        passing = passing & (etas < eta_cut)
        passing = passing & (ldg_trk_pts > ldg_trk_pt_cut)
        passing = passing & n_prongs == 1
        # passing = passing & isol

        content = ak.Array(passing).layout
        reshaped_output = ak.layout.ListOffsetArray64(input_shape, content)
        passed_result = np.array(reshaped_output.any())
        content_rtau = ak.Array(rtaus).layout
        reshaped_rtaus = ak.layout.ListOffsetArray64(input_shape, content_rtau)
        selected_taus_pts = ak.Array(df["Taus_pt"][passed_result])[reshaped_output[passed_result]][:, 0]
        selected_taus_etas = ak.Array(df["Taus_eta"][passed_result])[reshaped_output[passed_result]][:, 0]
        selected_taus_phis = ak.Array(df["Taus_phi"][passed_result])[reshaped_output[passed_result]][:, 0]
        selected_taus_e = ak.Array(df["Taus_e"][passed_result])[reshaped_output[passed_result]][:, 0]
        selected_taus_ldgchgtrkpt = ak.Array(df["Taus_lChTrkPt"][passed_result])[reshaped_output[passed_result]][:, 0]
        selected_taus_rtau = ak.Array(reshaped_rtaus[passed_result])[reshaped_output[passed_result]][:, 0]
        data = np.array([selected_taus_pts, selected_taus_etas, selected_taus_phis, selected_taus_e, selected_taus_rtau, selected_taus_ldgchgtrkpt]).T
        return_df = pd.DataFrame(data,
                                 columns=["Tau_pt", "Tau_eta", "Tau_phi", "Tau_e", "Tau_rtau", "Tau_lChTrkPt"])
        return passed_result, return_df

    def _jet_selection(self, df):
        pt_cut = 20.0           #80.0
        eta_cut = 4.7           #1.4


        input_shape = ak.Array(df["Jets_pt"]).layout.offsets
        counts = np.array([len(x) for x in df["Jets_pt"]])
        pts = np.concatenate(df["Jets_pt"].ravel()).ravel()
        etas = np.concatenate(df["Jets_eta"].abs().ravel()).ravel()
        ids = np.concatenate(df["Jets_IDloose"].ravel()).ravel()
        bjet_discr = np.concatenate(df["Jets_pfCombinedInclusiveSecondaryVertexV2BJetTags"].ravel()).ravel()

        passing = ids
        passing = passing & (pts >= pt_cut)
        passing = passing & (etas < eta_cut)

        bjets_passing = passing & (bjet_discr > 0.8484) & (etas < 2.4)

        content = ak.Array(passing).layout
        content_bjet = ak.Array(bjets_passing).layout

        reshaped_output = ak.layout.ListOffsetArray64(input_shape, content)
        reshaped_output_bjet = ak.layout.ListOffsetArray64(input_shape, content_bjet)
        passed_results = reshaped_output.any() & (counts >= 3) & reshaped_output_bjet.any()
        if np.sum(passed_results)==0:
            return passed_results, pd.DataFrame(None, columns=["Jets_pt", "Jets_eta", "Jets_phi", "Bjet_pt", "Bjet_eta", "Bjet_phi"])
        selected_jets_pts = ak.Array(df["Jets_pt"][passed_results])[reshaped_output[passed_results]][:, :4]
        selected_jets_etas = ak.Array(df["Jets_eta"][passed_results])[reshaped_output[passed_results]][:, :4]
        selected_jets_phis = ak.Array(df["Jets_phi"][passed_results])[reshaped_output[passed_results]][:, :4]
        selected_bjet_pt = ak.Array(df["Jets_pt"][passed_results])[reshaped_output_bjet[passed_results]][:, 0]
        selected_bjet_eta = ak.Array(df["Jets_eta"][passed_results])[reshaped_output_bjet[passed_results]][:, 0]
        selected_bjet_phi = ak.Array(df["Jets_phi"][passed_results])[reshaped_output_bjet[passed_results]][:, 0]

        return_df = pd.DataFrame(None, columns=["Jets_pt", "Jets_eta", "Jets_phi", "Bjet_pt", "Bjet_eta", "Bjet_phi"])
        # return_df["Jets_pt"] = np.array(ak.to_list(selected_jets_pts), dtype='O')
        # return_df["Jets_eta"] = np.array(ak.to_list(selected_jets_etas), dtype='O')
        # return_df["Jets_phi"] = np.array(ak.to_list(selected_jets_phis), dtype='O')
        return_df["Jets_pt"] = ak.to_list(selected_jets_pts)
        return_df["Jets_eta"] = ak.to_list(selected_jets_etas)
        return_df["Jets_phi"] = ak.to_list(selected_jets_phis)
        return_df["Bjet_pt"] = np.array(selected_bjet_pt)
        return_df["Bjet_eta"] = np.array(selected_bjet_eta)
        return_df["Bjet_phi"] = np.array(selected_bjet_phi)

        return passed_results, return_df

    def _pass_standard_selections(self, df):
        passing_indices = df["l1met"].values > 60.0 #80.0
        passing_indices = passing_indices & df["HLT_LooseIsoPFTau50_Trk30_eta2p1_MET90_vx"].values
        passing_indices = passing_indices & df["met_passed_filters"].values
        passing_indices = passing_indices & (df["nGoodOfflineVertices"].values > 1)
        passing_indices = passing_indices & self._electron_selection(df)
        passing_indices = passing_indices & self._muon_selection(df)
        return passing_indices



    def _electron_selection(self, df):
        input_shape = ak.Array(df["Electrons_pt"]).layout.offsets
        pts = np.concatenate(df["Electrons_pt"].ravel()).ravel()
        etas = np.concatenate(df["Electrons_eta"].abs().ravel()).ravel()
        mvas = np.concatenate(df["Electrons_MVA"].ravel()).ravel()
        isol = np.concatenate(df["Electrons_effAreaMiniIso"].ravel()).ravel()
        passing = pts >= 15.0
        passing = (etas < 2.5) & passing
        passing = passing & (
            ((mvas >= -0.041) & (etas <= 0.8)) | \
                ((mvas >= 0.383) & ((etas > 0.8) & (etas < 1.479))) | \
                                     ((mvas >= -0.515) & (etas >= 1.479)))
        vetoed = passing & (isol < 0.4)
        content = ak.Array(vetoed).layout
        reshaped_output = ak.layout.ListOffsetArray64(input_shape, content)
        veto_result = np.logical_not(reshaped_output.any())
        return veto_result

    def _muon_selection(self, df):
        input_shape = ak.Array(df["Muons_pt"]).layout.offsets
        pts = np.concatenate(df["Muons_pt"].ravel()).ravel()
        etas = np.concatenate(df["Muons_eta"].abs().ravel()).ravel()
        ids = np.concatenate(df["Muons_muIDLoose"].ravel()).ravel()
        isol = np.concatenate(df["Muons_effAreaMiniIso"].ravel()).ravel()

        passing = pts > 10.0
        passing = passing & (etas < 2.5)
        passing = passing & ids
        vetoed = passing & (isol < 0.4)
        content = ak.Array(vetoed).layout
        reshaped_output = ak.layout.ListOffsetArray64(input_shape, content)
        veto_result = np.logical_not(reshaped_output.any())
        return veto_result

    def _deltar_a_to_b(self, a_etas, a_phis, b_etas, b_phis):
        print("working on it")

def add_delta_phi_met(row, index_phi, index_metphi):
    met_phi = row[index_metphi]
    phis = row[index_phi]
    if(type(phis) is list):
        phis = np.array(phis)
    dphis = np.abs(phis-met_phi)
    if(type(phis) is not np.ndarray):
        dphis = dphis if (dphis < np.pi) else dphis - 2 * np.pi
    else:
        indices_less_than = dphis < np.pi
        dphis[~indices_less_than] = dphis[~indices_less_than] - 2 * np.pi
    return dphis