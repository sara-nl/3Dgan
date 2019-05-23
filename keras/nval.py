from os import path
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array
import time

from EcalEnergyGan import generator, discriminator
from EcalEnergyTrain_hvd import GetData

num_events=1000
latent = 200
gm = generator(latent)

gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
c.SetGrid()
gStyle.SetOptStat(0)
Eprof = TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 500)


gweight1='/scratch/04653/damianp/3DGAN_256w_64n_bs8_11_1_n16all_RMSprop_200_newwheel/generator_params_generator_epoch_024.hdf5'
gweights = [gweight1]
label = ['4n_16w_bs8']
scales = [1]
filename = 'ecal_ratio_multi.pdf'

#Get Actual Data
# d=h5py.File("/scratch/04653/damianp/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_2_9.h5",'r')
# X=np.array(d.get('ECAL')[0:num_events], np.float32)                             
# Y=np.array(d.get('target')[0:num_events][:,1], np.float32)
# X[X < 1e-6] = 0
# Y = Y
# Data = np.sum(X, axis=(1, 2, 3))
X, Y, Data = GetData("/scratch/04653/damianp/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_2_9.h5")
X = X[:num_events]
Y = Y[:num_events]
Data = Data[:num_events]

for j in np.arange(num_events):
  Eprof.Fill(Y[j], Data[j]/Y[j])
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
Eprof.GetYaxis().SetRangeUser(0, 0.03)
color =2
Eprof.SetLineColor(color)
legend = TLegend(0.8, 0.8, 0.9, 0.9)
legend.AddEntry(Eprof, "Data", "l")
Gprof = []

import time 
for i, gweight in enumerate(gweights):
#for i in np.arange(1):
#   gweight=gweights[i]                                                                                                                                                                     
  Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), 100, 0, 500))
  #Gprof[i].SetStates(0)
  #Generate events
  gm.load_weights(gweight)
  noise = np.random.normal(0, 1, (num_events, latent))
  generator_in = np.multiply(Y, noise)
  start = time.time()
  generated_images = gm.predict(generator_in, verbose=False, batch_size=100)
  print "Time {}s".format(time.time() - start)
  print generated_images
  GData = np.sum(generated_images, axis=(1, 2, 3))/scales[i]

  print GData.shape
  for j in range(num_events):
    Gprof[i].Fill(Y[j], GData[j]/Y[j])
  color = color + 2
  Gprof[i].SetLineColor(color)
  Gprof[i].Draw('sames')
  c.Modified()
  legend.AddEntry(Gprof[i], label[i], "l")
  legend.Draw()
  c.Update()

c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)
# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
