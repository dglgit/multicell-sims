import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numba
from random import choices, expovariate
@numba.njit()
def weighted_prob(arr,p,s=1):
    #assuming sum(p)==1
    return arr[np.searchsorted(np.cumsum(p),np.random.random()*s,side='right')]
import xml.etree.ElementTree as ET
from collections import OrderedDict as odict
import numpy as np
import re


#compounds,changes,stoichs,rates,v_assoc,rv_assoc=parseVCML('SF_testing1.vcml')
#rates=adjust_constants(rates,stoichs,rv_assoc)
#print(rates)
#compounds['Deg_GFPmRNA']=100
#print(compounds.keys())

from numba import int32, float32, float64 , int64   # import the types
from numba.experimental import jitclass
cell_spec=[
    ('species',float64[:]),
    ('changes',float64[:,:]),
    ('stoichs',float64[:,:]),
    ('rates',float64[:]),
    ('assoc_v',float64[:])
]
@jitclass(cell_spec)
class ICell:
    def __init__(self, species, changes,stoichs,rates,assoc_v):
        self.species=species
        self.changes=changes
        self.stoichs=stoichs
        self.rates=rates
        self.assoc_v=assoc_v
from tqdm import tqdm
import multiprocessing as mp
import time
#@numba.njit()

@numba.njit()
def gillespieStepWithIndex(species, changes, law_coeffs,law_idxs,rates):
    #law_idxs=[np.argwhere(row>0).reshape(-1) for row in laws]
    #law_coeffs=[row[i,law_idxs] for i in range(len(laws))]
    propensities=np.empty(changes.shape[0],np.float64)
    for i in range(len(propensities)):
        propensities[i]=np.prod(species[law_idxs[i]]**law_coeffs[i])
    propensities*=rates
    summed=propensities.sum()
    action=weighted_prob(changes,p=propensities,s=summed)
    dt=expovariate(summed)
    return action,dt

@numba.njit(parallel=True)
def stacked_multicellular_idx_based(cells,shared_idx,timestep,t0,tend,maxSteps=float('inf')):
    #assumed shared_idx is the same across all cells 
    #assuming all cells are the same 
    #shared species will go at the front of the new species array and be taken out of their normal spots for each cell 
    #laws and stoichs will have their shared_idx removed and transferred to the start
    ts=np.arange(t0,tend,timestep)
    record=np.ones(shape=(len(cells),len(ts),cells[0].species.shape[0]))
    nonshared=np.array([i for i in range(len(cells[0].species)) if i not in shared_idx])
    steps=(tend-t0)//timestep
    nshared=len(shared_idx)
    totalSpecies=nshared+len(cells)*(len(cells[0].species)-nshared)
    totalRxns=len(cells[0].changes)*len(cells)
    allSpecies=np.empty(totalSpecies)
    allLaws=np.zeros(shape=(totalRxns,totalSpecies))
    allRates=np.empty(shape=totalRxns)
    allChanges=np.zeros(shape=(totalRxns,totalSpecies))
    sidx=nshared
    ridx=0
    allSpecies[:sidx]=cells[0].species[shared_idx]
    for cell in cells:
        cellSpecies=cell.species
        cellRxns=cell.changes
        nextSidx=sidx+len(nonshared)
        allSpecies[sidx:nextSidx]=cellSpecies[nonshared]
        allLaws[ridx:ridx+cellRxns.shape[0],sidx:nextSidx]=cell.stoichs[:,nonshared]
        allLaws[ridx:ridx+cellRxns.shape[0],:nshared]=cell.stoichs[:,shared_idx]
        allChanges[ridx:ridx+cellRxns.shape[0],sidx:nextSidx]=cell.changes[:,nonshared]
        allChanges[ridx:ridx+cellRxns.shape[0],:nshared]=cell.changes[:,shared_idx]
        allRates[ridx:ridx+cellRxns.shape[0]]=cell.rates
        sidx=nextSidx
        ridx+=cellRxns.shape[0]
    allLawIdxs=[np.argwhere(allLaws[i]).flatten() for i in range(len(allLaws))]
    allLawCoeffs=[allLaws[i,allLawIdxs[i]] for i in range(len(allLaws))]
    tcount=0
    steps=0
    t=0
    for i in range(len(ts)):
        while t//timestep<i+1:
            ds,dt=gillespieStepWithIndex(allSpecies,allChanges,allLawCoeffs,allLawIdxs,allRates)
            tcount+=dt
            t+=dt
            allSpecies+=ds
            steps+=1
            if steps>maxSteps:
                print('max steps exceeded')
                return ts[:i],record[:,:i,:]
            #print(f'i={i}, t={t}, tcount={tcount}, increment={t//timestep}')
        tcount=0
        for j in range(len(cells)):
            nspecies=len(cells[j].species)
            record[j,i,nshared:]=allSpecies[nshared+j*len(nonshared):nshared+(j+1)*len(nonshared)]
        record[:,i,:nshared]=allSpecies[:nshared]
    #print("done with sim in",steps,"steps, about", float(steps/len(ts)),"/",timestep," or ",float(steps/timestep),"/second")
    return ts,record
def clampSpecies(changes,idx):
    a=changes.copy()
    a[idx]=0
    return a
def makeReplicate(nCells,species,changes,stoichs,rates, shared_idx,timestep,t0,tend, numReplicate=10):
    #runs simulations many times 
    #returns times and 4d array of records of shape [trial number, cell number, time number, species]
    #take mean per cell record[:,:, -1 , speciesIdx].mean(1)
    ts=np.arange(t0,tend,timestep)
    allRecord=np.empty((numReplicate,nCells,len(ts),len(species)),np.float64)
    populations=[[ICell(species.copy(),changes.copy(),stoichs.copy(),rates.copy(),np.ones(len(stoichs),np.float64)) 
                  for i in range(nCells)] for j in range(numReplicate)]
    for i in tqdm(range(numReplicate)):
    #for i in numba.prange(numReplicate):
    #for i in range(numReplicate):
        cells=populations[i]
        _,trialRecord=stacked_multicellular_idx_based(cells,shared_idx,timestep,t0,tend)
        allRecord[i,:,:,:]=trialRecord
    print('done')
    return ts,allRecord

def fixVCML(fname):
    xmlStr=''
    with open(fname,'r') as f:
        xmlStr=''.join(list(f))
    xmlStr=re.sub(' xmlns="[^"]+"', '', xmlStr, count=1)
    tree=ET.fromstring(xmlStr)

    return tree
def parseModel(model_xml):
    compounds=odict()
    assoc_v=odict()
    rxns=[]
    stoich=[]
    for child in model_xml:
        if child.tag=='LocalizedCompound':
            compounds[child.attrib['Name']]=0
            assoc_v[child.attrib['Name']]=child.attrib['Structure']
    return compounds, assoc_v
def parseCompartmentJumps(compartment_xml,compounds):
    rxn_jump=odict()#keyed to reaction names:jump vector
    rxn_stoich=odict()
    compound_list=list(compounds.keys())
    for c in compartment_xml:
        if c.tag=='JumpProcess':
            rxn_jump[c.attrib['Name']]=np.zeros(len(compounds))
            rxn_stoich[c.attrib['Name']]=np.zeros(len(compounds))
            for i in c:
                potentialHidden=[]
                if i.tag=='Effect':
                    idx=list(compounds.keys()).index(i.attrib['VarName'].replace('_Count',''))
                    coeff=int(float(i.text))
                    if coeff==0:
                        potentialHidden.append(idx)
                    elif coeff<0:
                        rxn_stoich[c.attrib['Name']][idx]=-coeff
                    rxn_jump[c.attrib['Name']][idx]=coeff
                if not (rxn_jump[c.attrib['Name']]<0).any():
                    for idx in potentialHidden:
                        rxn_stoich[c.attrib['Name']][idx]=1
    return rxn_jump,rxn_stoich



def adjust_constants(rates, stoichs, rxn_assoc_v,override_v=5000):
    #assumes rates are in uM if there is more than one
    kmole=0.0016605387#represents 1/(s*molecules) i think
    #molecules = 10^-6 * C uM * 10^-15 * V um^3 *6.022*10^23 = C * 6.022*10^2 * V
    #k [=] 1/(s*uM^(order-1))=>1/(s*(molecules)^(order-1))
    #k in uM * uM 1/(s* 6.022*10^2 * V)
    orders=stoichs.sum(-1)
    return rates * 1/((6.022*100*rxn_assoc_v)**(orders-1))
def parseVCML(fname):
    root=fixVCML(fname)
    constants = odict()#keyed to names
    compounds=odict() #keyed to compound names:concentration
    rxn_stoichs=odict()#keyed to reaction names:stoich vector
    rate_constants=odict()#keyed to reaction names
    compartments=odict()#keyed to compartment names
    rxn_jumps=odict()
    compound_assoc_v=odict()#volumes of the compartments each compound is in, 1d
    rxn_assoc_v=odict()#volumes that each reaction takes place, 1d
    rxn_ks=0
    print(list(root[0]))
    for child in root[0]:
        name=child.tag
        if child.tag=='Model':
            compounds,compound_assoc_v=parseModel(child)#compound assoc_v is just names for rn
        elif child.tag=='SimulationSpec':
            print(child.attrib)
            if child.attrib['Stochastic']=='true':
                for child2 in child:
                    if child2.tag=='MathDescription':
                        for c in child2:
                            if c.tag=='Constant':
                                try:
                                    constants[c.attrib['Name']]=float(c.text)
                                except ValueError as ve:
                                    print(c.text,c.attrib['Name'],ve)
                                    pass
                            elif c.tag=='CompartmentSubDomain':
                                rxn_jumps,rxn_stoichs=parseCompartmentJumps(c,compounds)
                                #print(rxn_jumps)
                                rxn_ks=[constants[f"Kr_{i[:-8]}"] if 'reverse' in i else constants[f'Kf_{i}'] for i in rxn_jumps]
            if len(constants)>0:
                for v in compound_assoc_v:
                    name=compound_assoc_v[v]
                    #print(constants)
                    compound_assoc_v[v]=constants['Size_'+name]
                break


    for name in compounds: #fills conc arr with number of molecules
        #compounds[name]=np.round(constants[name+'_Count_init_uM']*compound_assoc_v[name]*6.022*100)
        compounds[name]=constants[name+'_Count_initCount']
    
    #rxn_ks['Kr__3OC6HSLdiffusion']*=compound_assoc_v['_3OC6HSL_inside']/compound_assoc_v['_3OC6HSL_out']#idk why its like this but yeah
    compound_names=list(compounds.keys())
    rxn_names=list(rxn_stoichs)
    constant_names=list(constants)
    rxn_ks[3]*=constants['Size_Environment']/constants['Size_Cell']
    conc_arr=np.array(list(compounds.values()))
    jump_arr=np.array([rxn_jumps[i] for i in rxn_jumps])
    stoich_arr=np.array(list(rxn_stoichs.values()))
    rate_arr=np.array(rxn_ks)
    v_arr=np.array(list(compound_assoc_v.values()))
    for r in rxn_stoichs:
        idx=np.where(rxn_stoichs[r]>0)[0][0]
        #print(np.where(rxn_stoichs[r]>0))
        rxn_assoc_v[r]=compound_assoc_v[compound_names[idx]]
    rxn_v_arr=np.array(list(rxn_assoc_v.values()))
    adjusted=adjust_constants(rxn_ks,stoich_arr,rxn_v_arr)
    for i in range(len(rxn_ks)):
        print(adjusted[i],list(rxn_stoichs)[i],i)
    return compounds,jump_arr,stoich_arr,rate_arr,v_arr,rxn_v_arr

