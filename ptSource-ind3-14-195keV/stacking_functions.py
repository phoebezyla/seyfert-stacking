import os, sys, time
import pickle
from astropy import units as u
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

import astromodels
from astromodels import clone_model
import threeML

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from threeML import *
    from threeML.plugins.experimental.CastroLike import *
    from hawc_hal import HAL, HealpixConeROI, HealpixMapROI

OMP_NUM_THREADS = 1
MKL_NUM_THREADS = 1
NUMEXPR_NUM_THREADS = 1

def plot_logProfile(IntC,xbest,minllh,show=False,save=None):
    x = np.logspace(-30,-8,200)
    totalllh = np.zeros(200)

    for i,fn in enumerate(x):
        for j,cont in enumerate(IntC):
            totalllh[i] += cont(fn)

    plt.plot(x,-totalllh,'b+')
    plt.hlines(minllh+2.71/2,x[0],x[-1],color='red',ls='--')
    plt.xscale('log')
    plt.ylim(minllh-1,minllh+10)
    plt.xlim(x[0],x[-1])
    fun_bin = interp1d(-totalllh[x>np.power(10,xbest)],x[x>np.power(10,xbest)]) #only intersted in UL for now
    
    if show:
        plt.show()
    if save is not None:
        fig.savefig("{}".format(save))
    fig.clear()
    plt.close(fig)
    
def plot_logProfile_alt(IntC,param_df,like_df,name,minlogN=-30.,maxlogN=-1.,show=False,save=None):
    finalnorm = np.linspace(minlogN,maxlogN,200)
    finalnorm = np.power(10,finalnorm)
    totalllh = np.zeros(200)
    
    for i,fn in enumerate(finalnorm):
        for j,cont in enumerate(IntC):
            totalllh[i] += cont(fn)
    
    llhinterp = InterpolatedUnivariateSpline(np.log10(finalnorm),totalllh,k=1,ext=0)
    minNorm = param_df['value'][0]#np.power(10,res.x)
    minLLH = like_df.iloc[1]['-log(likelihood)']
    fig,sbu = plt.subplots()
    plt.plot(finalnorm,-totalllh-minLLH,'b',markersize=2) #totallh needs a min, to make it positive
    #plt.plot(finalnorm,-totalllh,'b',markersize=2)
    #plt.ylabel('-totalllh')
    plt.vlines(minNorm,0.,3.0,linestyles='--')
    plt.xscale('log')
    plt.title("Likelihood profile for %s"%(name))
    #plt.ylim(0,2.71)
    #plt.xlim(np.power(10,minlogN),np.power(10,maxlogN))
    plt.xlabel("Normalization [kev-1 s-1 cm-2]")
    plt.ylabel("LLH-LLHmin")
    plt.grid()
    
    if show:
        plt.show()
    if save is not None:
        fig.savefig("{}".format(save))
    fig.clear()

def saveResults(llh,jl,name,pivot,index):
    jointRes = jl.results
    jointRes.optimized_model.save("model_files/yml_ind%s_optimized/E_%.1f_TeV/%s_fit_mod.yml"%(index,pivot,name),overwrite=True)
    #jointRes.optimized_model.save("models/fitted_ix%s_%.1fTeV.yml"%(index,pivot),overwrite=True)

def plotResults(llh,jl,name):
    ## Model in counts spacve and residuals
    fig1 = llh.display_spectrum()
    fig1.savefig("plots/residuals/%s_res.png"%(name))

    ## Spectrum fit
    fig2 = plot_spectra(jl.results)
    plt.xlabel("Energy [TeV]")
    plt.ylabel(r"$E^2\,dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    plt.title("Spectrum fit for %s"%(name))
    fig2.savefig("plots/spectra/%s_fit_spectrum.png"%(name))

    ## Energy planes (model, datqa, residuals)
    fig3 = llh.display_fit(smoothing_kernel_sigma=0.3,display_colorbar=True)
    fig3.savefig("plots/energyplanes/%s_fit_planes.png"%(name))


def get_log_like_weighted(self):
    log_l = 0.0
    all_yy = np.split(
        self._likelihood_model.get_total_flux(self._all_xx), self._splits
    )
    for i, interval_container in enumerate(self._active_containers):
        xx = self._all_xx_split[i]
        yy = all_yy[i]
        length = interval_container.stop - interval_container.start
        expected_flux = scipy.integrate.simps(yy, xx) / length
        weight = getattr(interval_container, "weight", 1.0)   # use weights 
        this_log_l = interval_container(weight * expected_flux)
        log_l += this_log_l
    return -log_l   # convert -logL back to logL for threeML's convention



class StackingAnalysis():
    def __init__(self,intervalContainers):
        self.IntC = intervalContainers
        self.cl = None
        self.clm = None
        self.data = None
        self.model_source = None

    def calc_likelihoods(model, name, ra, dec, bins, MAP, DR, data_radius=5., model_radius=8.):
        roi = HealpixConeROI(data_radius=data_radius,
                         model_radius=model_radius,
                         ra=ra,
                         dec=dec)
        
        llh = HAL("Likelihood_{}".format(name),MAP,DR,roi)
        llh.set_active_measurements(bin_list=bins)
        llh.display()

        datalist = DataList(llh)
        
        jl = JointLikelihood(model,datalist,verbose=False)
        
        return llh, jl

    def likelihood_profile(indminNorm,lh,param_df,like_df,name,valN=200,computeTS=True):
        #norms = np.linspace(np.log10(indminNorm)-5,np.log10(indminNorm)+5,valN)
        
        normMin = -35
        normMax = -15
        norms = np.linspace(normMin,normMax,valN)
        log_val = np.zeros(valN)
        
        for j in range(valN):
            lh.verbose=False
            log_val[j] = lh.minus_log_like_profile(norms[j]) 

        norms = np.power(10,norms)

        if computeTS:
            a = lh.compute_TS(name,like_df)
            print(a)
        else: 
            a = "TS not computed through lh.compute_TS()"
            print(a)

        return norms, log_val, a

#    def stacked_likelihood(IntC,clm):
#        cl = CastroLike("stacked",IntC)
#        cl.set_model(clm)
#        data = DataList(cl)
#        
#        fjl = JointLikelihood(clm,data,verbose=False)
#        return fjl, data
        
    def ptsource_model(name,ra,dec,A,pivot,ind=-3.0,Kmax=1e-3):
        spectrum = Powerlaw()
        spectrum.index = ind
        spectrum.index.free = False
        spectrum.K.unit= (u.keV * u.s * u.cm**2 )**(-1)
        spectrum.K = 1e-21 * A
        spectrum.K.min_value = 1e-35
        spectrum.K.max_value = Kmax
        spectrum.K.free = True
        #spectrum.K.transformation = log10
        spectrum.piv = pivot
        spectrum.piv.free = False
        spectrum.piv.unit = u.TeV
        source = PointSource(name,ra,dec,spectrum)
        
        source.position.ra.free = False
        source.position.dec.free = False
        model = Model(source)

        return source, model
    
    def perform_bayesian_analysis_CI(model_source, clm, data, mide, UB=1e-12,nW=10,nB=150,nS=3000):
        model_source.spectrum.main.Powerlaw.K.prior = Uniform_prior(lower_bound=0.0,upper_bound=UB)
        ba = BayesianAnalysis(clm, data)
        ba.set_sampler("emcee")
        
        ba.sampler.setup(nS,n_burn_in=nB,n_walkers=nW)
        res = ba.sample()
        results = ba.results.get_variates('finalNorm.spectrum.main.Powerlaw.K')
        samples = results.samples
        
        credInt = np.quantile(samples,[0.5,0.95,0.159,0.84])
        results.append(credInt[0]*1e9*Atotal) # 50% CI Norm [TeV-1 s-1 cm-2]
        resultsHigh.append(credInt[1]*1e9*Atotal) # 95% CI Norm 
        resultsLow.append(credInt[2]*1e9*Atotal) # 16% CI Norm

        return credInt, results, resultsHigh, resultsLow

    def bayesian_ana(model_source, clm, data, mide, DIR, sourceName, Atotal, UB=1e-12,nW=10,nB=150,nS=3000):
        model_source.spectrum.main.Powerlaw.K.prior = Uniform_prior(lower_bound=0.0,upper_bound=UB)
        ba = BayesianAnalysis(clm, data)
        ba.set_sampler("emcee")
        ba.sampler.setup(n_walkers=nW,n_burn_in=nB,n_iterations=nS)
        ba.sample()
        
        samples = ba.samples
        samples_file = os.path.join(DIR,'bayes_res','{}_bayes_{}TeV.csv'.format(sourceName,mide))
        
        with open(samples_file,'wb') as f:
            pickle.dump(samples,f)

        print('\n*****Bayesian results*****')
        credInt = np.quantile(samples['finalNorm.spectrum.main.Powerlaw.K'],0.95)
        uplim.append(credInt*1e9*Atotal)
        print('95% CI Norm: {} [TeV-1 s-1 cm-2]'.format(uplim[-1]))
        
        return credInt,uplim

