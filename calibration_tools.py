import mastcasjobs
from astropy.io import ascii
from astropy.table import Table

from uncertainties import ufloat
from uncertainties.umath import *

import sys
import os
import re
import numpy as np
import pylab
import json
import matplotlib.pylab as plt
from matplotlib import gridspec
import pandas as pd 
import math

from uncertainties import ufloat
from uncertainties.umath import *

from astropy.stats import sigma_clip

import sigmacut

try: # Python 3.x
	from urllib.parse import quote as urlencode
	from urllib.request import urlretrieve
except ImportError:  # Python 2.x
	from urllib import pathname2url as urlencode
	from urllib import urlretrieve

try: # Python 3.x
	import http.client as httplib 
except ImportError:  # Python 2.x
	import httplib

# get the WSID and password if not already defined
import getpass
if not os.environ.get('CASJOBS_WSID'):
	os.environ['CASJOBS_WSID'] = '734534650'#input('Enter Casjobs WSID:')
if not os.environ.get('CASJOBS_PW'):
	os.environ['CASJOBS_PW'] = 'Dragon12435'


import warnings
warnings.filterwarnings("ignore")


jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")

def PS1_mean_detections(objids):
	j = 0
	if len(objids) == 0:
		objids = [objids]
	for obj in objids:

		queryMean = """select 
		objID, raMean, decMean, gMeanPSFMag, gMeanPSFMagErr, 
		rMeanPSFMag, rMeanPSFMagErr, 
		iMeanPSFMag, iMeanPSFMagErr, 
		zMeanPSFMag, zMeanPSFMagErr
		from (
		select * from MeanObjectView where objID={}
		) d
		""".format(obj)
		
		meanResults = jobs.quick(queryMean, task_name="python mean table search")
		
		data = [list(i) for i in meanResults]
		columns = list(meanResults.columns)

		df = pd.DataFrame(data=data,columns=columns)
		df = df.replace(-999,np.nan)
		if j == 0:
			all_df = df
			j += 1
		else:
			all_df = all_df.append(df)

	return all_df
	

def PS1_detections(objids):
	
	j = 0
	if len(objids) == 0:
		objids = [objids]
	for obj in objids:

		queryDetections = """ select
		objID, detectID, filter=f.filterType, obsTime, ra, dec,
		psfFlux, psfFluxErr, zp, psfLikelihood, psfQf, psfChiSq, psfMajorFWHM, psfMinorFWHM, psfQfPerfect, 
		apFlux, apFluxErr, telluricExt, infoFlag, infoFlag2, infoFlag3, airMass
		from (
		select * from Detection where objID={}
		) d
		join Filter f on d.filterID=f.filterID
		order by d.filterID, obsTime
		""".format(obj)
		
		detectionResults = jobs.quick(queryDetections, task_name="python detections table search")
		
		data = [list(i) for i in detectionResults]
		columns = list(detectionResults.columns)
		
		df = pd.DataFrame(data=data,columns=columns)
		df = df.replace(-999,np.nan)
		if j == 0:
			all_df = df
			j += 1
		else:
			all_df = all_df.append(df)
	
	return all_df 


def Mean_calc(data,err,sigma=3):
	calcaverage = sigmacut.calcaverageclass()
	#calcaverage.calcaverage_sigmacutloop(data)
	calcaverage.calcaverage_sigmacutloop(data,noise=err,Nsigma=sigma,median_firstiteration=True,saveused=True)
	return calcaverage.mean, calcaverage.mean_err, ~calcaverage.clipped


def mag_error(flux,fluxerr):
	mag = -2.5 * np.log10(flux) + 8.9
	mag_err = (2.5 / np.log(10)) * (fluxerr / flux)
	return mag, mag_err

def clip_detections(mag,mag_err):
	mask = ~sigma_clip(mag, sigma=3, maxiters=5,masked=True).mask
	mag = mag[mask]
	mag_err = mag_err[mask]
	return mag, mag_err

def correct_err(mag_err,correction):
	mag_err = np.sqrt(mag_err**2 + correction**2)	
	return mag_err

def Error_sigma_clip(mag,mag_err,sigma=3,correction=0):
	if correction == 0:	
		mean, err, ind = Mean_calc(mag,None,sigma)
	else:
		mean, err, ind = Mean_calc(mag,mag_err,sigma)
	if type(mean) == type(None):
		mean = np.nan
		err = np.nan
	return mean, err, ind

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    numerator = sum([values[i]*weights[i] for i in range(0,len(values))])
    denominator = sum([i for i in weights])
    weighted_mean = numerator/denominator
    return weighted_mean


def Mean_mag(table,sigma,correction=0):
	df = pd.DataFrame(columns = ['objID','ra','dec','gMeanPSFMag','gMeanPSFMagErr',
					'rMeanPSFMag','rMeanPSFMagErr','iMeanPSFMag','iMeanPSFMagErr',
					'zMeanPSFMag','zMeanPSFMagErr','yMeanPSFMag','yMeanPSFMagErr',
					'gMeanApMag','gMeanApMagErr','rMeanApMag','rMeanApMagErr',
					'iMeanApMag','iMeanApMagErr','zMeanApMag','zMeanApMagErr',
					'yMeanApMag','yMeanApMagErr',
					'gStdPSFMag','gMedianPSFMagErr','gStdApMag','gMedianApMagErr',
					'rStdPSFMag','rMedianPSFMagErr','rStdApMag','rMedianApMagErr',
					'iStdPSFMag','iMedianPSFMagErr','iStdApMag','iMedianApMagErr',
					'zStdPSFMag','zMedianPSFMagErr','zStdApMag','zMedianApMagErr',
					'yStdPSFMag','yMedianPSFMagErr','yStdApMag','yMedianApMagErr',
					'gPSFChi2','rPSFChi2','iPSFChi2','zPSFChi2','yPSFChi2',
					'gApChi2','rApChi2','iApChi2','zApChi2','yApChi2'])
	ids = table['objID'].unique()
	for i in ids:
		obj = table.iloc[table['objID'].values == i]
		filts = table['filter'].unique()
		obj_df = pd.DataFrame(data = np.zeros((53,53))*np.nan,
					columns = ['objID','ra','dec','gMeanPSFMag','gMeanPSFMagErr',
					'rMeanPSFMag','rMeanPSFMagErr','iMeanPSFMag','iMeanPSFMagErr',
					'zMeanPSFMag','zMeanPSFMagErr','yMeanPSFMag','yMeanPSFMagErr',
					'gMeanApMag','gMeanApMagErr','rMeanApMag','rMeanApMagErr',
					'iMeanApMag','iMeanApMagErr','zMeanApMag','zMeanApMagErr',
					'yMeanApMag','yMeanApMagErr',
					'gStdPSFMag','gMedianPSFMagErr','gStdApMag','gMedianApMagErr',
					'rStdPSFMag','rMedianPSFMagErr','rStdApMag','rMedianApMagErr',
					'iStdPSFMag','iMedianPSFMagErr','iStdApMag','iMedianApMagErr',
					'zStdPSFMag','zMedianPSFMagErr','zStdApMag','zMedianApMagErr',
					'yStdPSFMag','yMedianPSFMagErr','yStdApMag','yMedianApMagErr',
					'gPSFChi2','rPSFChi2','iPSFChi2','zPSFChi2','yPSFChi2',
					'gApChi2','rApChi2','iApChi2','zApChi2','yApChi2'])
		obj_df = obj_df.iloc[0] # very gross "solution" to weird size error
		obj_df['objID'] = int(i)
		obj_df['ra'] = np.nanmedian(obj['ra'].values)
		obj_df['dec'] = np.nanmedian(obj['dec'].values)

		for f in filts:
			detections = obj.iloc[obj['filter'].values == f]
			psf_mag, psf_mag_err = mag_error(detections.psfFlux.values,
											detections.psfFluxErr.values)

			psf_mag, psf_mag_err = clip_detections(psf_mag,psf_mag_err)
			ind = (psf_mag == -999) | (psf_mag_err == -999)

			psf_mag[ind] = np.nan
			psf_mag_err[ind] = np.nan


			if (len(psf_mag) < 0.5*len(detections)) | (len(psf_mag) < 10):
				psf_mag[:] = np.nan 
				psf_mag_err[:] = np.nan

			psf_mag_err = correct_err(psf_mag_err,correction)


			psf_mean, psf_err, ind = Error_sigma_clip(psf_mag,psf_mag_err,sigma=3,correction=0)

			#psf_mag = psf_mag[ind]
			#psf_mag_err = psf_mag_err[ind]

			weight_method = weighted_avg_and_std([ufloat(psf_mag[i], psf_mag_err[i]) for i in range(0, len(psf_mag_err))],
                                                   [1/i for i in psf_mag_err])
			#psf_mean = weight_method.nominal_value
			median_psf_err = np.nanmedian(psf_mag_err[ind])#np.sqrt(np.nanmedian((2.5 / np.log(10)) * (detections.psfFluxErr.values 
							#								   / detections.psfFlux.values))**2 + correction **2)#psf_mag_err[ind])
			#median_psf_err = weight_method.std_dev
			std_psf_mags = np.nanstd(psf_mag[ind])

			red_chi2 = np.nansum((psf_mag[ind] - psf_mean)**2/(psf_mag_err[ind])**2) / (len(ind)-1)
			#if red_chi2 < 10:
			obj_df[f + 'MeanPSFMag'] = psf_mean
			obj_df[f + 'MeanPSFMagErr'] = psf_err
			obj_df[f + 'StdPSFMag'] = std_psf_mags #weight_method.std_dev
			obj_df[f + 'MedianPSFMagErr'] = median_psf_err
			obj_df[f + 'PSFChi2'] = red_chi2


			ap_mag, ap_mag_err = mag_error(detections.apFlux.values,
										detections.apFluxErr.values)
			
			ap_mag, ap_mag_err = clip_detections(ap_mag,ap_mag_err)
			ap_mag_err = correct_err(ap_mag_err, correction)
			
			ap_mean, ap_err, ind = Mean_calc(ap_mag,None,sigma)
			if type(ap_mean) == type(None):
				ap_mean = np.nan
			median_ap_err = np.nanmedian(ap_mag_err[ind])
			std_ap_mags = np.nanstd(ap_mag[ind])
	
			red_chi2 = np.nansum((ap_mag[ind] - ap_mean)**2/(ap_mag_err[ind])**2) / (len(ind)-1)

			obj_df[f + 'MeanApMag'] = ap_mean
			obj_df[f + 'MeanApMagErr'] = ap_err
			obj_df[f + 'StdApMag'] = std_ap_mags
			obj_df[f + 'MedianApMagErr'] = median_ap_err
			obj_df[f + 'ApChi2'] = red_chi2

		df = df.append(obj_df)
	return df


def PS1_lc(ID,filt,table,correction=0):
	columns = ['objID','ra','dec','obstime','psfFlux',
					'psfFluxErr','apFlux','apFluxErr']

	obj = table.iloc[table['objID'].values == ID]
	
	detections = obj.iloc[obj['filter'].values == filt]
	detections = detections[columns]
	detections['psfMag'] = -2.5 * np.log10(detections.psfFlux.values) + 8.9
	psf_mag_err = (-2.5 / np.log10(10)) * (detections.psfFluxErr.values / detections.psfFlux.values)
	
	detections['psfMagErr'] = np.sqrt(psf_mag_err**2 + correction**2)

	detections['apMag'] = -2.5 * np.log10(detections.apFlux.values) + 8.9
	ap_mag_err = (-2.5 / np.log10(10)) * (detections.apFluxErr.values / detections.apFlux.values)
	detections['apMagErr'] = np.sqrt(ap_mag_err**2 + correction**2)
	
	return detections



