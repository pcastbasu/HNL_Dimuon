import numpy as np
import pandas as pd
import os
import siren
from siren.SIREN_Controller import SIREN_Controller
import matplotlib.pyplot as plt
import collections

experiment      = "IceCube"
controller      = SIREN_Controller(1, experiment)
icecube_geo     = controller.fid_vol

def get_file_list(masses,mixings,deepcore=True,atmos=True,astro=True):
    #This function gives us the path to every simulated file
    filepaths = []
    if deepcore == True: 
        if atmos == True and astro == False:
            for i in masses:
                for j in mixings:
                    filepath = os.path.join('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pcastanobasurto/IceCube_Dimuon/march17sims/dc_atmosnumu/output%5.5e'%(float(i)*10**(-3)),'test%5.5e.parquet'%j)
                    if os.path.exists(filepath):
                        filepaths.append(filepath)
        elif atmos == False and astro == True: 
            for i in masses:
                for j in mixings:
                    filepath = os.path.join('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pcastanobasurto/IceCube_Dimuon/march17sims/dc_astronumu/output%5.5e'%(float(i)*10**(-3)),'test%5.5e.parquet'%j)
                    if os.path.exists(filepath):
                        filepaths.append(filepath)
        else:
            return -1      
    if deepcore == False: 
        if atmos == True and astro == False:
            for i in masses:
                for j in mixings:
                    filepath = os.path.join('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pcastanobasurto/IceCube_Dimuon/march17sims/atmosnumu/output%5.5e'%(float(i)*10**(-3)),'test%5.5e.parquet'%j)
                    if os.path.exists(filepath):
                        filepaths.append(filepath)
        elif atmos == False and astro == True: 
            for i in masses:
                for j in mixings:
                    filepath = os.path.join('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pcastanobasurto/IceCube_Dimuon/march17sims/astronumu/output%5.5e'%(float(i)*10**(-3)),'test%5.5e.parquet'%j)
                    if os.path.exists(filepath):
                        filepaths.append(filepath)
        else:
            return -1
    return filepaths

def get_data_from_file(file):
    df = pd.read_parquet(file)
    #first we find the incoming neutrino and primary hnl 4-momenta in the lab frame
    nu4momentum  = []
    hnl4momentum = []
    for i in df['primary_momentum']:
        nu4mom  = i[0]
        hnl4mom = i[1]
        nu4momentum.append(nu4mom)
        hnl4momentum.append(hnl4mom)
    #now we find the dimuon 4momenta in the lab frame
    muon4momentum     = []
    antimuon4momentum = []
    openingangles     = [] #in deg
    for i in df['secondary_momenta']:
        muon4mom     = i[1][1]
        antimuon4mom = i[1][2]
        muon4momentum.append(muon4mom)
        antimuon4momentum.append(antimuon4mom)
        muonmom     = muon4mom[1:4]
        antimuonmom = antimuon4mom[1:4]
        dotproduct  = np.dot(muonmom,antimuonmom)
        angle       = np.arccos(dotproduct/(np.linalg.norm(muonmom)*np.linalg.norm(antimuonmom)))
        openingangles.append(angle*180/np.pi)
    #last we find the weights and vertices
    weights = []
    for i in df['event_weight']:
        weights.append(i)
    np.array(weights)[np.isnan(np.array(weights))] = 0
    vertices = []
    for i in df['vertex']:
        vertices.append(i)
    
    return nu4momentum,hnl4momentum,muon4momentum,antimuon4momentum,openingangles,weights,vertices

def get_sep_dists(muon_vertices, muon_momenta, antimuon_momenta):
    S_maxs = []
    S_mins = []
    L_mins = []
    for i in range(len(muon_vertices)):
        muon_intersections     = icecube_geo.ComputeIntersections(siren.math.Vector3D(muon_vertices[i]),siren.math.Vector3D(muon_momenta[i]))
        antimuon_intersections = icecube_geo.ComputeIntersections(siren.math.Vector3D(muon_vertices[i]),siren.math.Vector3D(antimuon_momenta[i]))
        sepdists = []
        for elem1,elem2 in zip(muon_intersections,antimuon_intersections):
            if siren.math.Vector3D.magnitude(elem1.position - elem2.position)>0:
                sepdists.append(siren.math.Vector3D.magnitude(elem1.position - elem2.position))
        if sepdists:
            S_maxs.append(max(sepdists))
            S_mins.append(min(sepdists))
        else: 
            S_maxs.append(0)
            S_mins.append(0) 
        lengths = []
        if len(muon_intersections) == 0 and len(antimuon_intersections) == 0:
            L_mins.append(0)   
        elif len(muon_intersections) == 1 and len(antimuon_intersections) == 0:
            L_mins.append(siren.math.Vector3D.magnitude(muon_momenta[i].position - muon_intersections[0].position))   
        elif len(muon_intersections) == 0 and len(antimuon_intersections) == 1:
            L_mins.append(siren.math.Vector3D.magnitude(antimuon_momenta[i].position - antimuon_intersections[0].position))
        elif len(muon_intersections) == 1 and len(antimuon_intersections) == 1:
            lengths.append(siren.math.Vector3D.magnitude(antimuon_momenta[i].position - antimuon_intersections[0].position))
            lengths.append(siren.math.Vector3D.magnitude(muon_momenta[i].position - muon_intersections[0].position))
            L_mins.append(min(lengths))
        elif len(muon_intersections) == 2 and len(antimuon_intersections) == 1:
            lengths.append(siren.math.Vector3D.magnitude(antimuon_momenta[i].position - antimuon_intersections[0].position))
            lengths.append(siren.math.Vector3D.magnitude(muon_intersections[0].position - muon_intersections[1].position))
            L_mins.append(min(lengths))
        elif len(muon_intersections) == 1 and len(antimuon_intersections) == 2:
            lengths.append(siren.math.Vector3D.magnitude(muon_momenta[i].position - muon_intersections[0].position))
            lengths.append(siren.math.Vector3D.magnitude(antimuon_intersections[0].position - antimuon_intersections[1].position))
            L_mins.append(min(lengths))  
        elif len(muon_intersections) == 2 and len(antimuon_intersections) == 2:
            lengths.append(siren.math.Vector3D.magnitude(muon_intersections[0].position - muon_intersections[1].position))
            lengths.append(siren.math.Vector3D.magnitude(antimuon_intersections[0].position - antimuon_intersections[1].position))
            L_mins.append(min(lengths))
        else:
            L_mins.append(0)
        
    return S_maxs, S_mins, L_mins

def get_xs_ratio(m4,umu4,energies):
    cross_section_model = "HNLDISSplines"
    xsfiledir           = siren.utilities.get_cross_section_model_path(cross_section_model)
    primary_NuMu        = siren.dataclasses.Particle.ParticleType.NuMu
    primary_NuMuBar     = siren.dataclasses.Particle.ParticleType.NuMuBar
    target_type         = siren.dataclasses.Particle.ParticleType.Nucleon
    DIS_xs_NuMu         = siren.interactions.HNLDISFromSpline(
                                os.path.join(xsfiledir, "M_0000000MeV/dsdxdy-nu-N-nc-GRV98lo_patched_central.fits"),
                                os.path.join(xsfiledir, "M_%sMeV/sigma-nu-N-nc-GRV98lo_patched_central.fits"%m4),
                                float(m4)*1e-3, [0, umu4, 0], [primary_NuMu],[target_type],)
    DIS_xs_NuMuBar      = siren.interactions.HNLDISFromSpline(
                                os.path.join(xsfiledir, "M_0000000MeV/dsdxdy-nubar-N-nc-GRV98lo_patched_central.fits"),
                                os.path.join(xsfiledir, "M_%sMeV/sigma-nubar-N-nc-GRV98lo_patched_central.fits"%m4),
                                float(m4)*1e-3, [0, umu4, 0], [primary_NuMuBar],[target_type],)
    total_xs_NuMu       = np.array([DIS_xs_NuMu.TotalCrossSection(primary_NuMu, energy) for energy in energies])
    total_xs_NuMuBar    = np.array([DIS_xs_NuMuBar.TotalCrossSection(primary_NuMuBar, energy) for energy in energies])
    cross_section_ratio = total_xs_NuMu / total_xs_NuMuBar
    return cross_section_ratio

def make_num_dimuons_dict(filepaths, sep_cuts, ang_cuts, L_cuts, muon_energy_threshold, masses,mixings, mixings_norm,seconds,xs_ratio_weight):
    num_dimuons = {}
    for index,file in enumerate(filepaths):
        mass,mixing = float(masses[index//len(mixings)])*1e-3, mixings_norm[index%len(mixings)]**2
        #print(mass,mixing)
        key         = tuple((mass,mixing))
        print("%d/%d"%(index,len(filepaths)),end="\r")
        nu4momentum,hnl4momentum,muon4momentum,antimuon4momentum,openingangles,weights,vertices = get_data_from_file(file)
        nu_energies          = [nue[0] for nue in nu4momentum]
        muon_vertices        = [vtx[1] for vtx in vertices]
        muon_momenta         = [mumom[1:4] for mumom in muon4momentum]
        antimuon_momenta     = [amumom[1:4] for amumom in antimuon4momentum]
        S_maxs,S_mins,L_mins = get_sep_dists(muon_vertices, muon_momenta, antimuon_momenta)
        muon_energies        = [mue[0] for mue in muon4momentum] 
        muon_energy_flag     = np.array(muon_energies) > muon_energy_threshold
        weights_new          = np.where(muon_energy_flag,np.where(np.isnan(weights),0,weights),0)*seconds
        if xs_ratio_weight == True:
            weights_new  = np.array(weights_new)*get_xs_ratio(masses[index//len(mixings)],np.sqrt(mixing),np.array(nu_energies))
        num_dimuons[key] = {}
        for sep in sep_cuts:
            if "sep" not in num_dimuons[key]:
                num_dimuons[key]["sep"] = {}
            num_dimuons[key]["sep"][sep] = sum(weights_new[np.array(S_maxs) > sep])
        for ang in ang_cuts:
            if "ang" not in num_dimuons[key]:
                num_dimuons[key]["ang"] = {}
            num_dimuons[key]["ang"][ang] = sum(weights_new[np.array(openingangles) > ang])
        for dis in L_cuts:
            if "dis" not in num_dimuons[key]:
                num_dimuons[key]["dis"] = {}
            num_dimuons[key]["dis"][dis] = sum(weights_new[np.array(L_mins) > dis])
    return num_dimuons

