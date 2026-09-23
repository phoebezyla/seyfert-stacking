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

def saveResults(llh,jl,name,pivot,index):
    jointRes = jl.results
    #jointRes.optimized_model.save("model_files/yml_ind%s_optimized/E_%.1f_TeV/%s_fit.yml"%(index,pivot,name),overwrite=True)
    jointRes.optimized_model.save("models/fitted_ix%s_%.1fTeV.yml"%(index,pivot),overwrite=True)

def plotResults(llh,jl,name):
    ## Model in counts spacve and residuals
    fig1 = llh.display_spectrum()
    #fig1.savefig("plots/residuals/%s_res.png"%(name))
    fig1.savefig("plots/%s_res.png"%(name))

    ## Spectrum fit
    fig2 = plot_spectra(jl.results)
    plt.xlabel("Energy [TeV]")
    plt.ylabel(r"$E^2\,dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
    plt.title("Spectrum fit for %s"%(name))
    #fig2.savefig("plots/spectra/%s_fit_spectrum.png"%(name))
    fig2.savefig("plots/%s_fit_spectrum.png"%(name))

    ## Energy planes (model, datqa, residuals)
    fig3 = llh.display_fit(smoothing_kernel_sigma=0.3,display_colorbar=True)
    #fig3.savefig("plots/energyplanes/%s_fit_planes.png"%(name))
    fig3.savefig("plots/%s_fit_planes.png"%(name))


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


class PhoebePlotting():
    def PowerLaw(x,A,pivot,gamma): 
        return A * (x/pivot) ** (-gamma)

    def SpectrumPlots(dfs,figname,keys=None,ylims=None,E_low=0.5e9,E_high=100e9,
                      PIV=[1e9,5e9,10e9],IX=[2.0,2.7,3.0]):
        """
        Builds four plots: One for each index (with the three different pivot
        assumption curves), and one with all nine curves. Included on the plots
        is a dot at the best-fit normalization for each pivot energy.
 
        Dataframe input is a dict built from the outputs from a pd.read_csv of a stacked result csv
        The dict needs to have keys {key: dataframe} of the same length as IX
        """

        if keys is None:
            keys = list(dfs.keys())
        assert len(keys) == len(IX), "keys and inds must be the same length"

        xarr = np.logspace(np.log10(E_low),np.log10(E_high), 1000)

        c = plt.cm.tab10(np.linspace(0, 1, len(PIV))).reshape(len(PIV), 4) #One per pivot 
        ls = ['-', '--', ':']  # One per index
 
        # Individual per-index plots #
        for j, key in enumerate(keys):
            df = dfs[key]
            ix = IX[j]

            plt.figure(figsize=[10,8],layout='constrained')

            for i, E0 in enumerate(PIV):
                A_best = df.loc[i, 'indminNorm']
                yarr = PhoebePlotting.PowerLaw(xarr, A_best, E0, ix)
                plt.plot(xarr,yarr,color=c[i],
                    label=f"E0 = {E0:.1E} keV, A = {A_best:.2E}")
        
                plt.plot(E0, A_best, 'o', color=c[i],label="_nolabel_")
        
            if ylims is not None: 
                plt.ylim(ylims)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Energy [keV]")
            plt.ylabel(r"Flux [keV$^{-1}$ s$^{-1}$ cm$^{-2}$]")
            plt.title(f"Spectrum across pivot assumptions, Index = {ix}")
            plt.legend()
            plt.grid(True, which='both', alpha=0.3)
            plt.savefig(f"{figname}_ind{key}.png")
            plt.close()

        # Combined Plot #
        handles = {}
        fig, ax = plt.subplots(figsize=[10, 8], layout='constrained')
        
        for j, key in enumerate(keys):
            df = dfs[key]
            ix = IX[j]

            for i, E0 in enumerate(PIV):
                A_best = df.loc[i, 'indminNorm']
        
                yarr = PhoebePlotting.PowerLaw(xarr, A_best, E0, ix)
                line, = ax.plot(
                    xarr, yarr, color=c[i], ls=ls[j],
                    label=f"E0 = {E0:.1E} keV, Index = {ix}"
                )
                handles[(i,j)] = line
                ax.plot(E0, A_best, 'o', color=c[i])
        
        ordered_handles = [handles[(i,j)] for i in range(len(PIV)) for j in range(len(IX))]
        ordered_labels = [h.get_label() for h in ordered_handles]
        
        if ylims is not None:
            ax.set_ylim(1e-27,5e-17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel(r"Flux [keV$^{-1}$ s$^{-1}$ cm$^{-2}$]")
        ax.set_title("Spectrum curves: all indices and pivot assumptions")
        ax.legend(ordered_handles, ordered_labels, fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        
        fig.savefig(f"{figname}_combined.png")
        plt.close(fig)

    def LikelihoodPlots(dfs,figname,keys=None,xlims=None,PIV=[1e9,5e9,10e9],IX=[2.0,2.7,3.0]):
        """
        Builds four plots: One for each index (with the three pivot value curves),
        and one combined (nine curves). On the per-index plots, there is a vertical line
        indicating the best-fit normalization point. Same inputs as SpectrumPlot
        """

        if keys is None:
            keys = list(dfs.keys())
        assert len(keys) == len(IX), "keys and inds must be the same length"


        c = plt.cm.tab10(np.linspace(0, 1, len(PIV))).reshape(len(PIV), 4) #One per pivot 
        ls = ['-', '--', ':']  # One per index

        norms_cols = [f"norms_{k}"   for k in range(200)]
        log_cols   = [f"log_val_{k}" for k in range(200)]

        ## Per-index plots ##
        for j, key in enumerate(keys):
            df = dfs[key]
            ix = IX[j]

            fig, ax = plt.subplots(figsize=[10,8],layout='constrained')
            
            for i, row in df.iterrows():
                norms = row[norms_cols].values.astype(float)
                logs  = row[log_cols].values.astype(float)
         
                logs_shifted = logs - logs.min()  # minimum is at 0, because we're comparing curves
        
                ax.plot(norms, logs_shifted, color=c[i], label=f"E = {PIV[i]:.1E} keV")
                ax.axvline(row['indminNorm'],color=c[i], ls = ":", alpha = 0.6, label="Best-fit normalization")
            ax.axhline(0.5, color='r', ls = '--', alpha = 0.7, label=r"$\Delta$logL = 0.5")
        
            if xlims is not None:
                ax.set_xlim(xlims)
            ax.set_xscale('log')
            ax.set_xlabel(r"Normalization [keV$^{-1}$ s$^{-1}$ cm$^{-2}$]")
            ax.set_ylabel("-logL - min('logL)")
            ax.set_title(f"Likelihood Profiles, Index = {ix}")
            ax.legend()
    
            fig.savefig(f"{figname}_index{key}.png")
            plt.close(fig)
        
        ## Combined Likelihood Plot ##
        fig, ax = plt.subplots(figsize=[10,8],layout='constrained')
        handles = {}
        
        for j, key in enumerate(keys):
            df = dfs[key]
            ix = IX[j]

            for i, row in df.iterrows():
                norms = row[norms_cols].values.astype(float)
                logs  = row[log_cols].values.astype(float)
                logs_shifted = logs - logs.min()  # minimum is at 0, because we're comparing curves
            
                line, = ax.plot(norms, logs_shifted, color=c[i], 
                    label=f"Pivot = {PIV[i]:.1E} keV, Index = {ix}",
                    ls=ls[j])
                handles[(i,j)] = line
                ax.axvline(row['indminNorm'],color=c[i], ls = ":", alpha = 0.6, label="_nolabel_")
            
        ordered_handles = [handles[(i,j)] for i in range(len(PIV)) for j in range(len(IX))]
        ordered_labels = [h.get_label() for h in ordered_handles]
   
        ax.axhline(0.5, color='r', ls = '--', alpha = 0.7, label=r"$\Delta$logL = 0.5") 
        if xlims is not None:    
            ax.set_xlim(xlims)
        ax.set_xscale('log')
        ax.set_xlabel(r"Normalization [keV$^{-1}$ s$^{-1}$ cm$^{-2}$]")
        ax.set_ylabel("-logL - min('logL)")
        ax.set_title(f"Likelihood Profiles")
        ax.legend(ordered_handles, ordered_labels, fontsize=8)
            
        fig.savefig(f"{figname}_combined.png")
        plt.close(fig)
 
    def PlotsSimple(norms, logs):
        




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
        normMax = -10
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

