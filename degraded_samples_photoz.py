# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import GPz
from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pylab as pb



########### Model options ###############

method = 'VC'               # select method, options = GL, VL, GD, VD, GC and VC [required]
                            #
m = 25                      # number of basis functions to use [required]
                            #
joint = True                # jointly learn a prior linear mean function [default=true]
                            #
heteroscedastic = True      # learn a heteroscedastic noise process, set to false interested only in point estimates
                            #
csl_method = 'normal'       # cost-sensitive learning option: [default='normal']
                            #       'balanced':     to weigh rare samples more heavly during train
                            #       'normalized':   assigns an error cost for each sample = 1/(z+1)
                            #       'normal':       no weights assigned, all samples are equally important
                            #
binWidth = 0.1              # the width of the bin for 'balanced' cost-sensitive learning [default=range(z_spec)/100]

decorrelate = True          # preprocess the data using PCA [default=False]

########### Training options ###########

dataPath_samples = '/mnt/zfsusers/stylianou/project/samples.csv'    # path to the data set, has to be in the following format m_1,m_2,..,m_k,e_1,e_2,...,e_k,z_spec
                                        # where m_i is the i-th magnitude, e_i is its associated uncertainty and z_spec is the spectroscopic redshift
                                        # [required]
dataPath_degraded_samples = '/mnt/zfsusers/stylianou/project/degraded_samples.csv'


maxIter = 500                   # maximum number of iterations [default=200]
maxAttempts = 50              # maximum iterations to attempt if there is no progress on the validation set [default=infinity]
trainSplit = 0.2               # percentage of data to use for training
validSplit = 0.2               # percentage of data to use for validation
testSplit  = 0.6               # percentage of data to use for testing

########### Start of script ###########

# Load samples
# read data from file
data = loadtxt(open(dataPath_samples,"rb"),delimiter=",")

X_samples = data[:, 1:6]
n,d = X_samples.shape
Y_samples = data[:, 0].reshape(n, 1)

filters = d/2

# log the uncertainties of the magnitudes, any additional preprocessing should be placed here
X_samples[:, int(filters):] = log(X_samples[:, int(filters):])



# Load degraded_samples
# read data from file
data = loadtxt(open(dataPath_degraded_samples,"rb"),delimiter=",")

X_degraded_samples = data[:, 1:6]
n,d = X_degraded_samples.shape
Y_degraded_samples = data[:, 0].reshape(n, 1)

filters = d/2

# log the uncertainties of the magnitudes, any additional preprocessing should be placed here
X_degraded_samples[:, int(filters):] = log(X_degraded_samples[:, int(filters):])




# sample training, validation and testing sets from the data
training,validation,testing = GPz.sample(n,trainSplit,validSplit,testSplit)

# you can also select the size of each sample
# training,validation,testing = GPz.sample(n,10000,10000,10000)

# get the weights for cost-sensitive learning
omega = GPz.getOmega(Y_samples, method=csl_method)




# initialize the initial model
model_samples = GPz.GP(m,method=method,joint=joint,heteroscedastic=heteroscedastic,decorrelate=decorrelate)

# train the model
model_samples.train(X_samples.copy(), Y_samples.copy(), omega=omega, training=training, validation=validation, maxIter=maxIter, maxAttempts=maxAttempts)

########### NOTE ###########
# you can train the model gain, eve using different data, by executing:
# model.train(model,X,Y,options)

# use the model to generate predictions for the test set
mu_samples,sigma_samples,modelV_samples,noiseV_samples,_ = model_samples.predict(X_samples[testing,:].copy())




# initialize the initial model
model_degraded_samples = GPz.GP(m,method=method,joint=joint,heteroscedastic=heteroscedastic,decorrelate=decorrelate)

# train the model
model_degraded_samples.train(X_degraded_samples.copy(), Y_degraded_samples.copy(), omega=omega, training=training, validation=validation, maxIter=maxIter, maxAttempts=maxAttempts)

########### NOTE ###########
# you can train the model gain, eve using different data, by executing:
# model.train(model,X,Y,options)

# use the model to generate predictions for the test set
mu_degraded_samples,sigma_degraded_samples,modelV_degraded_samples,noiseV_degraded_samples,_ = model_degraded_samples.predict(X_degraded_samples[testing,:].copy())







########### Display Results ###########

# compute metrics   (compared to samples - true redshifts)
rmse_samples = sqrt(GPz.metrics(Y_samples[testing],mu_samples,sigma_samples,lambda y,mu,sigma: (y-mu)**2))
mll_samples  = GPz.metrics(Y_samples[testing],mu_samples,sigma_samples,lambda y,mu,sigma: -0.5*(y-mu)**2/sigma-0.5*log(sigma)-0.5*log(2*pi))
fr15_samples = GPz.metrics(Y_samples[testing],mu_samples,sigma_samples,lambda y,mu,sigma: 100.0*(abs(y-mu)/(y+1.0)<0.15))
fr05_samples = GPz.metrics(Y_samples[testing],mu_samples,sigma_samples,lambda y,mu,sigma: 100.0*(abs(y-mu)/(y+1.0)<0.05))
bias_samples = GPz.metrics(Y_samples[testing],mu_samples,sigma_samples,lambda y,mu,sigma: y-mu)



# compute metrics   (compared to samples - true redshifts)
rmse_degraded_samples = sqrt(GPz.metrics(Y_samples[testing],mu_degraded_samples,sigma_degraded_samples,lambda y,mu,sigma: (y-mu)**2))
mll_degraded_samples  = GPz.metrics(Y_samples[testing],mu_degraded_samples,sigma_degraded_samples,lambda y,mu,sigma: -0.5*(y-mu)**2/sigma-0.5*log(sigma)-0.5*log(2*pi))
fr15_degraded_samples = GPz.metrics(Y_samples[testing],mu_degraded_samples,sigma_degraded_samples,lambda y,mu,sigma: 100.0*(abs(y-mu)/(y+1.0)<0.15))
fr05_degraded_samples = GPz.metrics(Y_samples[testing],mu_degraded_samples,sigma_degraded_samples,lambda y,mu,sigma: 100.0*(abs(y-mu)/(y+1.0)<0.05))
bias_degraded_samples = GPz.metrics(Y_samples[testing],mu_degraded_samples,sigma_degraded_samples,lambda y,mu,sigma: y-mu)





# print metrics for the entire data
print(('{0:4s}\t\t\t{1:3s}\t\t\t{2:6s}\t\t\t{3:6s}\t\t\t{4:4s}'.format('RMSE', ' MLL', ' FR15', ' FR05', ' BIAS')))
print(('{0:1.7e}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}'.format(rmse_samples[-1], mll_samples[-1], fr15_samples[-1],fr05_samples[-1],bias_samples[-1])))
print(('{0:1.7e}\t{1: 1.7e}\t{2: 1.7e}\t{3: 1.7e}\t{4: 1.7e}'.format(rmse_degraded_samples[-1], mll_degraded_samples[-1], fr15_degraded_samples[-1],fr05_degraded_samples[-1],bias_degraded_samples[-1])))


# plot scatter plots for density and uncertainty
f = plt.figure(1)
#plt.scatter(Y[testing,:],mu,s=5,c=log(squeeze(sigma)), edgecolor='none')
plt.scatter(Y_samples[testing,:][0:100],mu_samples[0:100],s=5, edgecolor=['none'])
plt.scatter(Y_samples[testing,:][0:100],mu_degraded_samples[0:100],s=5, edgecolor=['none'])
#f.show()
#plt.show()
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Photometric Redshift')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_1.pdf")


f = plt.figure(2)
xy = hstack([Y_samples[testing,:],mu_samples]).T
xy = hstack([Y_samples[testing,:],mu_degraded_samples]).T
z = gaussian_kde(xy)(xy)
plt.scatter(Y_samples[testing,:],mu_samples,s=5, edgecolor=['none'])
plt.scatter(Y_samples[testing,:],mu_degraded_samples,s=5, edgecolor=['none'])
#c=z
#f.show()
#plt.show()
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Photometric Redshift')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_2.pdf")


# plot the change in metrics as functions of data percentage
x = array(list(range(0,20+1)))*5
x[0]=1

ind_s = x*len(rmse_samples) // 100
ind_ds = x*len(rmse_degraded_samples) // 100

f = plt.figure(3)
plt.plot(x,rmse_samples[ind_s-1],'o-')
plt.plot(x,rmse_degraded_samples[ind_ds-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('RMSE')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_3.pdf")


f = plt.figure(4)
plt.plot(x,mll_samples[ind_s-1],'o-')
plt.plot(x,mll_degraded_samples[ind_ds-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('MLL')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_4.pdf")


f = plt.figure(5)
plt.plot(x,fr15_samples[ind_s-1],'o-')
plt.plot(x,fr15_degraded_samples[ind_ds-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('FR15')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_5.pdf")


f = plt.figure(6)
plt.plot(x,fr05_samples[ind_s-1],'o-')
plt.plot(x,fr05_degraded_samples[ind_ds-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('FR05')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_6.pdf")


f = plt.figure(7)
plt.plot(x,bias_samples[ind_s-1],'o-')
plt.plot(x,bias_degraded_samples[ind_ds-1],'o-')
plt.xlabel('Percentage of Data')
plt.ylabel('BIAS')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_7.pdf")


# plot mean and standard deviation of different scores as functions of spectroscopic redshift using 20 bins
f = plt.figure(8)
centers_s,means_s,stds_s = GPz.bin(Y_samples[testing],Y_samples[testing]-mu_samples,20)
centers_ds,means_ds,stds_ds = GPz.bin(Y_samples[testing],Y_samples[testing]-mu_degraded_samples,20)
plt.errorbar(centers_s,means_s,stds_s,fmt='o')
plt.errorbar(centers_ds,means_ds,stds_ds,fmt='o')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Bias')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_8.pdf")


f = plt.figure(9)
centers_s,means_s,stds_s = GPz.bin(Y_samples[testing],sqrt(modelV_samples),20)
centers_ds,means_ds,stds_ds = GPz.bin(Y_samples[testing],sqrt(modelV_degraded_samples),20)
plt.errorbar(centers_s,means_s,stds_s,fmt='o')
plt.errorbar(centers_ds,means_ds,stds_ds,fmt='o')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Model Uncertainty')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_9.pdf")

f = plt.figure(10)
centers_s,means_s,stds_s = GPz.bin(Y_samples[testing],sqrt(noiseV_samples),20)
centers_ds,means_ds,stds_ds = GPz.bin(Y_samples[testing],sqrt(noiseV_degraded_samples),20)
plt.errorbar(centers_s,means_s,stds_s,fmt='o')
plt.errorbar(centers_ds,means_ds,stds_ds,fmt='o')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel('Noise Uncertainty')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
#f.show()
#plt.show()
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_10.pdf")



# plot mu_samples against mu_degraded_samples
f = plt.figure(11)
plt.scatter(mu_samples, mu_degraded_samples, s=5)
plt.xlabel('Photometric Redshift of Samples')
plt.ylabel('Photometric Redshift of Degraded Samples')
plt.title('500 iterations - Trained on Degraded Samples and Tested on Samples')
pb.savefig("/mnt/zfsusers/stylianou/project/figures/degraded_samples_11.pdf")


# save output as a comma seperated values (mean,sigma,model_variance,noise_variance)
savetxt(method+'_'+str(m)+'_'+csl_method+'.csv', array([mu_samples,sigma_samples,modelV_samples,noiseV_samples])[:,:,0].T, delimiter=',')

savetxt(method+'_'+str(m)+'_'+csl_method+'.csv', array([mu_degraded_samples,sigma_degraded_samples,modelV_degraded_samples,noiseV_degraded_samples])[:,:,0].T, delimiter=',')

