import awkward as ak
import numpy as np
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory #, TreeMakerSchema
import treemaker
from treemaker import TreeMakerSchema
import fastjet
from lund_utils.Utils import get_quantile_axis2

import sys, os
sys.path.insert(0, '')
sys.path.append("../")
from lund_utils.Utils import *
import math

def run_reweighting(year, constituents, jets, nDarkHadronsPerJet, nProngsPerJet, nevts_tot, minPt = 0):
    # Lund systematic setup
    nToys = 100
    nSys = 10

    nevts_batch = nevts_tot

    LP_weights = np.zeros(nevts_batch)
    LP_mjj_check = np.zeros(nevts_batch)
    LP_weights_stat_var = np.zeros((nevts_batch, nToys))
    LP_weights_pt_var = np.zeros((nevts_batch, nToys))
    LP_weights_sys_var = np.zeros((nevts_batch, nSys))

    f_ratio_current = ROOT.TFile.Open("../LundReweighting/data/ratio_"+str(year)+".root")

    LP_rws = LundReweighter( f_ratio = f_ratio_current, min_pt = minPt )

    #Noise used to generated smeared ratio's based on stat unc
    np.random.seed(123)
    rand_noise = np.random.normal(size = (nToys, LP_rws.h_ratio.GetNbinsX(), LP_rws.h_ratio.GetNbinsY(), LP_rws.h_ratio.GetNbinsZ()))
    pt_rand_noise = np.random.normal(size = (nToys, LP_rws.h_ratio.GetNbinsY(), LP_rws.h_ratio.GetNbinsZ(), 3))

    out = LP_rws.get_all_weights(constituents, None, jets, do_sys_weights = True, distortion_sys = True, rand_noise = rand_noise, pt_rand_noise = pt_rand_noise, normalize = True, pf_cands_PtEtaPhiE_format = True, nDark = nDarkHadronsPerJet)

    return out

def prep_events(events):
    # mask to only svj categories
    mask = events["JetsAK8"].genIndex >= 0
    gen_jet_index = events["JetsAK8"].genIndex[mask]

    genJets = events["GenJetsAK8"][gen_jet_index]

    hv_category = genJets.hvCategory
    maskhv = (hv_category == 3) | (hv_category == 9) | (hv_category == 11) | (hv_category == 5) | (hv_category == 7) | (hv_category == 13)

    ak8genjets = genJets[maskhv]
    ak8jets = events["JetsAK8"]
    ak8jets_svj = ak8jets[maskhv]

    # print(str(len(ak8jets_svj)) + " events in sample")
    # print(str(len(ak.flatten(ak.nan_to_num(ak8jets.pt, nan=0)))) + " matched signal jets in sample.")

    maskMatched = ak8jets_svj.darkHadronJets.constituentsAssignedSecond != 1
    # print(ak.sum(ak8jets_svj.darkHadronJets.constituentsAssignedSecond), " number of constituents not used (assigned to second closest dark hadron.")

    ak8jets_constituents_pdgid = ak8jets_svj.darkHadronJets.constituentsPdgid[maskMatched]
    ak8jets_constituents = ak8jets_svj.darkHadronJets.constituents[maskMatched]

    constituents_pt = ak.nan_to_num(ak8jets_constituents.pt,nan=0)
    constituents_eta = ak.nan_to_num(ak8jets_constituents.eta,nan=0)
    constituents_phi = ak.nan_to_num(ak8jets_constituents.phi,nan=0)
    constituents_E = ak.nan_to_num(ak8jets_constituents.E,nan=0)

    jets_pt = ak.nan_to_num(ak8jets_svj.pt ,nan=0)
    jets_eta = ak.nan_to_num(ak8jets_svj.eta ,nan=0)
    jets_phi = ak.nan_to_num(ak8jets_svj.phi ,nan=0)
    jets_E = ak.nan_to_num(ak8jets_svj.E ,nan=0)

    #flatten to get a 1D list of the darkHadronJet constituents and the jets- will have to deal with postprocessing this later
    jetsFlat = ak.zip([ak.flatten(jets_pt), ak.flatten(jets_eta), ak.flatten(jets_phi), ak.flatten(jets_E) ] )
    constituentsFlat = ak.zip( [ak.flatten(ak.flatten(constituents_pt)), ak.flatten(ak.flatten(constituents_eta)), ak.flatten(ak.flatten(constituents_phi)), ak.flatten(ak.flatten(constituents_E)) ] )
    return constituentsFlat, jetsFlat, constituents_pt, jets_pt

def get_sumw_perProng(norm_dict, k, lund_weights, n_prongs):
    #separate norm per each number of prongs (so dist is not biased)
    if(n_prongs is None): n_prongs = np.ones_like(lund_weights, dtype=np.int32)
    max_prongs = int(round(np.amax(n_prongs)))

    lund_weights = np.clip(lund_weights, 0.1, 10.0)
    for n in range(1, max_prongs+1):
        key = k + "_" + str(n)
        mask = (n_prongs == n)
        weights = lund_weights[mask]
        if(len(weights) == 0): continue

        nJets = len(weights)
        sumw = 0
        if(len(weights.shape) > 1):
            sumw = np.sum(weights, axis = 0)
        else: sumw = np.sum(weights)

        norm_dict[key + "_sumw"] = sumw
        norm_dict[key + "_njets"] = nJets

    return norm_dict

def calculate_lund_weights(events, mc_year, subjetMinPt = 0):

    # number of events in current sample
    nevts_tot = ak.size(events["EvtNum"])
    #print(str(nevts_tot) + " events in sample")

    constituents, jets, constituents_pt, jets_pt = prep_events(events)

    nDarkHadronsPerJet = ak.num(constituents_pt,axis=2)
    nProngsPerJet = nDarkHadronsPerJet*2
    nJetsPerEvent = ak.num(jets_pt, axis=1)
    nProngsPerEvent = ak.flatten(nProngsPerJet)
    # print("expected njets  with lund reweighting", ak.sum(nJetsPerEvent))
    # print("nDarkHadrons", ak.sum(nDarkHadronsPerJet))
    # print("total number of prongs: ", ak.sum(nDarkHadronsPerJet)*2)
    # print("dark hadrons per jet", nDarkHadronsPerJet)

    # returns a weight per jet
    output = run_reweighting(mc_year, constituents, jets, nDarkHadronsPerJet, nProngsPerJet, nevts_tot, minPt = subjetMinPt)

    norm_dict = {}
    for k in output.keys():
        if len(output[k]) == 0: continue
        if k in ['bad_match', 'reclust_still_bad_match', 'reclust_nom', 'reclust_prongs_up', 'reclust_prongs_down', 'bquark_up', 'bquark_down', 'unclust_up', 'unclust_down', 'prongs_up', 'prongs_down']: continue
        elif k in ['subjetPts', 'subjetStatVars', 'subjetWeights', 'nSplittings', 'splittingWeights', 'lpIdxs', 'RawDistortionNoNorm', 'n_prongs', 'nomNoNorm']:
            newk = 'lundWeight' + k.capitalize().replace('_','').replace('vars', 'Vars').replace('up', 'Up').replace('down', 'Down').replace('distortion', 'Distortion').replace('nonorm', 'NoNorm')
            events[newk] = ak.unflatten(output[k], nJetsPerEvent)
        else:
            newk = 'lundWeight' + k.capitalize().replace('_','').replace('vars', 'Vars').replace('up', 'Up').replace('down', 'Down').replace('distortion', 'Distortion')
            output[k] = np.clip(output[k], 0.1, 10.0)
            norm_dict = get_sumw_perProng(norm_dict, newk, output[k], output['n_prongs'])
            events[newk] = ak.unflatten(output[k], nJetsPerEvent)
    # print("norm_dict", norm_dict)
    return events, norm_dict

def lund_normalization(events, field, norm, nJetsPerEvent):
    weights = ak.to_numpy(ak.flatten(events[field]))
    nProngs = list(map(str,ak.flatten(events['lundWeightNprongs'])))

    sumw  = np.array([norm[field+"_"+ p +"_sumw"] for p in nProngs])
    njets = np.array([norm[field+"_"+ p +"_njets"] for p in nProngs])

    if(len(weights.shape) > 1): weights_mean = sumw / njets[:,None]
    else: weights_mean = sumw / njets

    weights_mean = np.where(weights_mean == 0, 1, weights_mean)
    weights = weights / weights_mean
    events[field] = ak.unflatten(weights, nJetsPerEvent)

    return events[field]


def lund_post(events, field):
    if 'Nprongs' in field or 'Nsplittings' in field:
        events[field] = events[field]
    elif field == 'subjetStatVars':
        sub_stat_vars = ak.flatten(ak.flatten(events[field]), axis=-3)
        events[f"lundWeightSubJetStatVars"] = sub_stat_vars
        subweights_up = get_quantile_axis2(sub_stat_vars, 0.84)
        subweights_down = get_quantile_axis2(sub_stat_vars, 0.16)
        events[f"lundWeightSubJetStatUp"] = subweights_up
        events[f"lundWeightSubJetStatDown"] = subweights_down
    elif "StatVars" in field or "PtVars" in field:
        # Save Raw Jet toys
        stat_vars = events[field]
        events['lundWeightJetStatVars'] = stat_vars
        # Save jet level stat weights
        # Take the 16th and 84th percentiles. Awkward does not have a quantile function, so we have to do it by hand
        weights_up = get_quantile_axis2(stat_vars, 0.84)
        weights_down = get_quantile_axis2(stat_vars, 0.16)
        events[f"{field.replace('Vars', 'JetUp')}"] = weights_up
        events[f"{field.replace('Vars', 'JetDown')}"] = weights_down

        # Save Event level toys and weights
        event_weights = ak.prod(stat_vars, axis=-2)
        events['lundWeightEventStatVars'] = event_weights
        event_weights_up = get_quantile_axis1(event_weights, 0.84)
        event_weights_down = get_quantile_axis1(event_weights, 0.16)
        events[f"{field.replace('Vars', 'Up')}"] = np.clip(event_weights_up, 0,5)
        events[f"{field.replace('Vars', 'Down')}"] = np.clip(event_weights_down, 0,5)
    elif (field in ['lundWeightRawDistortion']):
        if 'RawDistortion' in field or 'Rawdistortion' in field or 'raw_distortion' in field:
            events["lundWeightJetRawDistortion"] = events[field]
        #Calculate Distortion Up and Down Variations

        ## Jet Level Variations
        # print('n negative raw distortion jets', ak.sum(events['lundWeightRawDistortion'] <0))
        # x = events['lundWeightRawDistortion'] - 1
        # events['lundWeightDistortionUp'] = events['lundWeightRawDistortion'] #events['lundWeightNom'] * events['lundWeightRawDistortion']
        # events['lundWeightDistortionDown'] = ak.where((1-x)<0, 0, (1-x)) #events['lundWeightNom'] * (1 - x)
        # events['lundWeightRawDistortion'] = ak.prod(events['lundWeightRawDistortion'], axis=-1)
        # evtDown = events['lundWeightDistortionDown']
        # events['lundWeightDistortionDown'] = ak.prod(events['lundWeightDistortionDown'], axis=-1)
        # maskrd = (events['lundWeightDistortionDown'] > 2)

        # print("down > 2: ", events['lundWeightRawDistortion'][maskrd][2:5])
        # print('x for down > 2: ', x[maskrd][2:5])
        # print('up for down > 2: ', events['lundWeightDistortionUp'][maskrd][2:5])
        # print('down for down > 2: ', evtDown[maskrd][2:5])

        # events['lundWeightDistortionUp'] = ak.prod(events['lundWeightDistortionUp'], axis=-1)

        # events['lundWeightRawDistortion'] = np.clip(events['lundWeightRawDistortion'], 0, 5)
        # events['lundWeightDistortionUp'] = np.clip(events['lundWeightDistortionUp'], 0, 5)
        # events['lundWeightDistortionDown'] = np.clip(events['lundWeightDistortionDown'], 0, 5)

        ### Event Level Variations
        events['lundWeightRawDistortion'] = ak.prod(events['lundWeightRawDistortion'], axis=-1)*events['lundWeightNom']
        x = events['lundWeightRawDistortion'] - 1
        events['lundWeightDistortionUp'] = events['lundWeightRawDistortion']
        events['lundWeightDistortionDown'] = ak.where((1-x)<0, 0, (1-x))

        events['lundWeightRawDistortion'] = np.clip(events['lundWeightRawDistortion'], 0, 5)
        events['lundWeightDistortionUp'] = np.clip(events['lundWeightDistortionUp'], 0, 5)
        events['lundWeightDistortionDown'] = np.clip(events['lundWeightDistortionDown'], 0, 5)

    elif (field in ['lundWeightNom', 'lundWeightSysUp', 'lundWeightSysDown', 'lundWeightNomNoNorm', 'lundWeightRawdistortionNoNorm']):
        events[field] = np.clip(ak.prod(events[field], axis=-1), 0, 5)
