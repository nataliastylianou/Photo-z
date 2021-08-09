# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import pylab as pb
from pzflow.examples import example_flow
from rail.creation import Creator, engines
from rail.creation.degradation import InvRedshiftIncompleteness, LineConfusion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import csv

flow = engines.FlowEngine(example_flow())
creator = Creator(flow)
degraded_creator = Creator(flow, degrader=InvRedshiftIncompleteness(0.8))


samples = creator.sample(100000)
degraded_samples = degraded_creator.sample(100000)

fig, ax = plt.subplots(figsize=(4.5,4.5), dpi=100)
ax.hist(samples['redshift'], bins=20, range=(0,2.3), histtype='step', label="Unbiased samples")
ax.hist(degraded_samples['redshift'], bins=20, range=(0,2.3), histtype='step', label="Degraded samples")
ax.legend()
ax.set(xlabel="Redshift", ylabel="Number of galaxies", xlim=(0,2.3))
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degr_1.pdf")


np.savetxt('/mnt/zfsusers/stylianou/project/samples.csv',samples, delimiter=',')

combined_pdfs=np.genfromtxt('/mnt/zfsusers/stylianou/project/samples.csv', delimiter=',')




degraded_samples.shape == samples.shape

def OxygenLineConfusion(data, seed=None):
    OII = 3727
    OIII = 5007
    
    data = LineConfusion(true_wavelen=OII, wrong_wavelen=OIII, frac_wrong=0.02)(data, seed)
    data = LineConfusion(true_wavelen=OIII, wrong_wavelen=OII, frac_wrong=0.01)(data, seed)
    return data

flow = engines.FlowEngine(example_flow())
creator = Creator(flow)
degraded_creator = Creator(flow, degrader=OxygenLineConfusion)

samples = creator.sample(100000, seed=0)
degraded_samples = degraded_creator.sample(100000, seed=0)

fig, ax = plt.subplots(figsize=(4.5,4.5), dpi=100)
ax.scatter(samples["redshift"], degraded_samples["redshift"], s=0.1)
ax.set(xlabel="True spec-z", ylabel="Erroneous spec-z")
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degr_2.pdf")

np.savetxt('/mnt/zfsusers/stylianou/project/degraded_samples.csv', degraded_samples, delimiter=',')

combined_pdfs=np.genfromtxt('/mnt/zfsusers/stylianou/project/degraded_samples.csv', delimiter=',')


















