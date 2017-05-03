#!/bin/env python
import ROOT
import logging
import numpy as np
import random
import csv
import os
import time


#hadronic decays
mytrainregfilehad = open('train_reg_ditau_mass_had_1e6.csv','wb')
mytestregfilehad = open('test_reg_ditau_mass_had_1e6.csv','wb')
myregfilehad = open('reg_ditau_mass_had_1e6.csv','wb')

writer_train_had = csv.writer(mytrainregfilehad)
writer_test_had = csv.writer(mytestregfilehad)
writer_had = csv.writer(myregfilehad)

mytrainregfileinclnuhad = open('train_reg_ditau_mass_inclnu_had_1e6.csv','wb')
mytestregfileinclnuhad = open('test_reg_ditau_mass_inclnu_had_1e6.csv','wb')
myregfileinclnuhad = open('reg_ditau_mass_inclnu_had_1e6.csv','wb')

writer_train_inclnu_had = csv.writer(mytrainregfileinclnuhad)
writer_test_inclnu_had = csv.writer(mytestregfileinclnuhad)
writer_inclnu_had = csv.writer(myregfileinclnuhad)

mytrainregfiletotalhad = open('train_reg_ditau_mass_total_had_1e6.csv','wb')
mytestregfiletotalhad = open('test_reg_ditau_mass_total_had_1e6.csv','wb')
myregfiletotalhad = open('reg_ditau_mass_total_had_1e6.csv','wb')

writer_train_total_had = csv.writer(mytrainregfiletotalhad)
writer_test_total_had = csv.writer(mytestregfiletotalhad)
writer_total_had = csv.writer(myregfiletotalhad)

mytrainregfilevishad = open('train_reg_ditau_vismass_had_1e6_new.csv','wb')
mytestregfilevishad = open('test_reg_ditau_vismass_had_1e6_new.csv','wb')
myregfilevishad = open('reg_ditau_vismass_had_1e6_new.csv','wb')

writer_vismass_had = csv.writer(myregfilevishad)
writer_train_vismass_had = csv.writer(mytrainregfilevishad)
writer_test_vismass_had = csv.writer(mytestregfilevishad)

#all decays
mytrainregfileall = open('train_reg_ditau_mass_all_1e6.csv','wb')
mytestregfileall = open('test_reg_ditau_mass_all_1e6.csv','wb')
myregfileall = open('reg_ditau_mass_all_1e6.csv','wb')

writer_train_all = csv.writer(mytrainregfileall)
writer_test_all = csv.writer(mytestregfileall)
writer_all = csv.writer(myregfileall)

mytrainregfileinclnu = open('train_reg_ditau_mass_inclnu_1e6.csv','wb')
mytestregfileinclnu = open('test_reg_ditau_mass_inclnu_1e6.csv','wb')
myregfileinclnu = open('reg_ditau_mass_inclnu_1e6.csv','wb')

writer_train_inclnu = csv.writer(mytrainregfileinclnu)
writer_test_inclnu = csv.writer(mytestregfileinclnu)
writer_inclnu = csv.writer(myregfileinclnu)

mytrainregfiletotal = open('train_reg_ditau_mass_total_1e6.csv','wb')
mytestregfiletotal = open('test_reg_ditau_mass_total_1e6.csv','wb')
myregfiletotal = open('reg_ditau_mass_total_1e6.csv','wb')

writer_train_total = csv.writer(mytrainregfiletotal)
writer_test_total = csv.writer(mytestregfiletotal)
writer_total = csv.writer(myregfiletotal)

mytrainregfiletheta = open('train_reg_ditau_theta_1e6.csv','wb')
mytestregfiletheta = open('test_reg_ditau_theta_1e6.csv','wb')
myregfiletheta = open('reg_ditau_theta_1e6.csv','wb')

writer_train_theta = csv.writer(mytrainregfiletheta)
writer_test_theta = csv.writer(mytestregfiletheta)
writer_theta = csv.writer(myregfiletheta)

mytrainregfilevisall = open('train_reg_ditau_vismass_all_1e6.csv','wb')
mytestregfilevisall = open('test_reg_ditau_vismass_all_1e6.csv','wb')
myregfilevisall = open('reg_ditau_vismass_all_1e6.csv','wb')

writer_vismass_all = csv.writer(myregfilevisall)
writer_train_vismass_all = csv.writer(mytrainregfilevisall)
writer_test_vismass_all = csv.writer(mytestregfilevisall)


ROOT.gSystem.Load('libDelphes')
ROOT.gInterpreter.Declare('#include "Delphes-3.4.0/classes/DelphesClasses.h"')

numberofevents = 1300000

#some basic logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)


chain = ROOT.TChain("Delphes")

for i in range(0,800):
    filepath = "/pnfs/psi.ch/cms/trivcat/store/user/clange/ztautau/ztautau_%s.root" % (i)
    myfile = "dcap://t3se01.psi.ch:22125//pnfs/psi.ch/cms/trivcat/store/user/clange/ztautau/ztautau_%s.root" % (i)
    if os.path.isfile(filepath) and os.stat(filepath).st_size > 700000000:
        logging.debug("getChain:Adding " + myfile)
        chain.AddFile(myfile)
        sampleEvents = chain.GetEntries()
        logger.info("Events: %d" % sampleEvents)
    else:
        continue

#start event loop
selectedEvents = 0
numberoftau = 0
wrongtaus = 0
normalcount = 0
anormalcount = 0
time_per_event = 0

for currentEvent, event in enumerate(chain):
    start = time.time()
    if (currentEvent % 5000 == 0):
        logger.info("Event {} of {}".format(currentEvent, sampleEvents))
    tau = []
    taue = []
    taumu = []
    taudp3 = []
    taudp4 = []
    taudp5 = []
    taudp6 = []
    taudp7 = []
    rest = [] 
    tauleptonic = []
    tauhadronic =[]
    motheroftau = []
    motleptonic = []
    motsemileptonic = []
    mothadronic = []
    
    for genPartIndex, genPart in enumerate(event.Particle): 
        if abs(genPart.PID) == 15:
            tau.append(genPartIndex)
            numberoftau +=1
    tauremove = []

    for j in tau:
        for genPartIndex, genPart in enumerate(event.Particle):
            if j in (genPart.D1, genPart.D2) and abs(genPart.PID) == 15:
                tauremove.append(genPartIndex)

    for k in tauremove:
        tau.remove(k)

    for u in tau:
        for genPartIndex, genPart in enumerate(event.Particle):
            if u in (genPart.M1, genPart.M2)and abs(genPart.PID) in (11,13):
                tauleptonic.append(u)
            elif u in (genPart.M1, genPart.M2) and abs(genPart.PID) not in (12,14,15,16,22):
                if u not in tauhadronic:
                    tauhadronic.append(u)
    for r in tau:
        for genPartIndex, genPart in enumerate(event.Particle):
            if r in (genPart.D1, genPart.D2) and abs(genPart.PID) != 15:
                if not genPartIndex in motheroftau: 
                    motheroftau.append(genPartIndex)

    for d in tauleptonic:
        for genPartIndex, genPart in enumerate(event.Particle):
            if d in (genPart.D1,genPart.D2) and abs(genPart.PID) !=15:
                if genPartIndex not in motleptonic:
                    motleptonic.append(genPartIndex)

    for i in tauhadronic:
        for genPartIndex, genPart in enumerate(event.Particle):
            if i in (genPart.D1, genPart.D2) and abs(genPart.PID) !=15 and genPartIndex in motleptonic:
                motleptonic.remove(genPartIndex)
                if genPartIndex not in motsemileptonic:
                    motsemileptonic.append(genPartIndex)
            elif i in (genPart.D1, genPart.D2) and abs(genPart.PID) !=15:
                if genPartIndex not in mothadronic:
                    mothadronic.append(genPartIndex)
    
    for j in motheroftau:
        v_vis_mot = ROOT.TLorentzVector()
        v_nu_mot = ROOT.TLorentzVector()
        metx = 0
        mety = 0
        tauofmot = []
        v_mot = ROOT.TLorentzVector()
        for genPartIndex,genPart in enumerate(event.Particle):
            if j in (genPart.M1,genPart.M2):
                tauofmot.append(genPartIndex)
        for i in tauofmot:
            nupx = 0
            nupy = 0
            for genPartIndex,genPart in enumerate(event.Particle):
                if i in (genPart.M1,genPart.M2):
                    v_mot += ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                    if abs(genPart.PID) not in (12,14,16):
                        v_vis_mot+= ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                    elif abs(genPart.PID) in (12,14,16):
                        v_nu_mot+= ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                        nupx += genPart.Px
                        nupy += genPart.Py
            metx +=nupx
            mety +=nupy       
        met = np.sqrt(metx*metx+mety*mety)
        writer_all.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),met,metx,mety,v_mot.M()))
        writer_inclnu.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),v_nu_mot.Px(),v_nu_mot.Py(),v_nu_mot.Pz(),v_nu_mot.E(),v_mot.M()))
        writer_total.writerow((v_mot.Px(),v_mot.Py(),v_mot.Pz(),v_mot.E(),v_mot.M()))
        writer_theta.writerow((v_mot.Px(),v_mot.Py(),v_mot.Pz(),v_mot.E(),v_mot.Theta()))
        if currentEvent < numberofevents*(7.0/10.0):
            writer_train_all.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),met,metx,mety,v_mot.M()))
            writer_train_inclnu.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),v_nu_mot.Px(),v_nu_mot.Py(),v_nu_mot.Pz(),v_nu_mot.E(),v_mot.M()))
            writer_train_total.writerow((v_mot.Px(),v_mot.Py(),v_mot.Pz(),v_mot.E(),v_mot.M()))
            writer_train_theta.writerow((v_mot.Px(),v_mot.Py(),v_mot.Pz(),v_mot.E(),v_mot.Theta()))
            writer_train_vismass_all.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),v_vis_mot.M()))
        elif currentEvent > numberofevents*(7.0/10.0):
            writer_test_all.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),met,metx,mety,v_mot.M()))
            writer_test_inclnu.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),v_nu_mot.Px(),v_nu_mot.Py(),v_nu_mot.Pz(),v_nu_mot.E(),v_mot.M()))
            writer_test_total.writerow((v_mot.Px(),v_mot.Py(),v_mot.Pz(),v_mot.E(),v_mot.M())) 
            writer_test_theta.writerow((v_mot.Px(),v_mot.Py(),v_mot.Pz(),v_mot.E(),v_mot.Theta())) 
            writer_test_vismass_all.writerow((v_vis_mot.Px(),v_vis_mot.Py(),v_vis_mot.Pz(),v_vis_mot.E(),v_vis_mot.M()))
    
          
    for p in mothadronic:
        tauofmothadronic = []
        for genPartIndex,genPart in enumerate(event.Particle):
            if p in (genPart.M1,genPart.M2):
                tauofmothadronic.append(genPartIndex)
        if len(tauofmothadronic) < 2:
            continue
        motheroftau_vis = ROOT.TLorentzVector()
        v_motheroftau = ROOT.TLorentzVector()
        v_nu_mot = ROOT.TLorentzVector()
        metx = 0
        mety = 0
        for g in tauofmothadronic:
            nupx = 0
            nupy = 0
            nuphi = 0
            decaycount = 0
            tau_vis = ROOT.TLorentzVector()
            for genPartIndex,genPart in enumerate(event.Particle):
                if g in (genPart.M1,genPart.M2) and (genPart.PID==111 or abs(genPart.PID)==211):
                    motheroftau_vis += ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                    tau_vis += ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                elif g in (genPart.M1,genPart.M2) and abs(genPart.PID)==16:
                    nupx += genPart.Px
                    nupy += genPart.Py
                    v_nu = ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                    v_nu_mot += ROOT.TLorentzVector(genPart.Px,genPart.Py,genPart.Pz,genPart.E)
                    decaycount +=1
            metx += nupx
            mety += nupy
            v_motheroftau += tau_vis + v_nu
        if decaycount is not 0:
            met = np.sqrt(metx*metx+mety*mety)
            writer_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),met,metx,mety,v_motheroftau.M()))
            writer_inclnu_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),v_nu_mot.Px(),v_nu_mot.Py(),v_nu_mot.Pz(),v_nu_mot.E(),v_motheroftau.M()))
            writer_total_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),met,metx,mety,v_motheroftau.M())) 
            if currentEvent < numberofevents*(7.0/10.0):
                writer_train_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),met,metx,mety,v_motheroftau.M()))
                writer_train_vismass_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),motheroftau_vis.M()))
                writer_train_inclnu_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),v_nu_mot.Px(),v_nu_mot.Py(),v_nu_mot.Pz(),v_nu_mot.E(),v_motheroftau.M()))
                writer_train_total_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),met,metx,mety,v_motheroftau.M())) 
            elif currentEvent > numberofevents*(7.0/10.0):
                writer_test_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),met,metx,mety,v_motheroftau.M()))
                writer_test_vismass_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),motheroftau_vis.M()))
                writer_test_inclnu_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),v_nu_mot.Px(),v_nu_mot.Py(),v_nu_mot.Pz(),v_nu_mot.E(),v_motheroftau.M()))
                writer_test_total_had.writerow((motheroftau_vis.Px(),motheroftau_vis.Py(),motheroftau_vis.Pz(),motheroftau_vis.E(),met,metx,mety,v_motheroftau.M()))
    
    end = time.time()
    time_per_event += end-start
    if currentEvent == numberofevents:
        break
    else:
        continue

print "execution time per event:", time_per_event/currentEvent
           
#hadronic decays
myregfilehad.close()
mytrainregfilehad.close()
mytestregfilehad.close()

myregfileinclnuhad.close()
mytrainregfileinclnuhad.close()
mytestregfileinclnuhad.close()

myregfiletotalhad.close()
mytrainregfiletotalhad.close()
mytestregfiletotalhad.close()

myregfilevishad.close()
mytrainregfilevishad.close()
mytestregfilevishad.close()

#all decays
myregfileall.close()
mytrainregfileall.close()
mytestregfileall.close()

myregfileinclnu.close()
mytrainregfileinclnu.close()
mytestregfileinclnu.close()

myregfiletotal.close()
mytrainregfiletotal.close()
mytestregfiletotal.close()

myregfiletheta.close()
mytrainregfiletheta.close()
mytestregfiletheta.close()

myregfilevisall.close()
mytrainregfilevisall.close()
mytestregfilevisall.close()

