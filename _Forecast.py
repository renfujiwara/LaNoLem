import numpy as np
import sys
import model.tool as tl
import time
from numpy import linalg as la
from model.nlds import NLDS
import argparse


DBG=tl.YES
#------------------------------------------------#
# I/O
MSAVE=tl.YES # save model.png (default: YES)
TSAVE=tl.NO # save every time tick (default: NO)
UPDATE_MDB=tl.YES    # update model DB or not (default: YES)
# --- windows 
RR_R=3         # lc=3*ls     --- len(Xc) current window length
LP_R=(1.0/5.0) # lp=ls*(1/5) --- sliding/reporting-window 
# --- LMfit 
W_RR='linear'  # weighted LMfit for RegimeReader (default: linear) 
W_RE='linear'  # weighted LMfit for RegimeEstimator (default: linear)
# --- model DB
MAXR=100           # max # of regimes (len(MDBs))
# --- RegimeEstimator
RE_TH_R=0.5            # th of insertion  e.g., th=0.5*|Xc|
RE_OVERLAP_LEN_R=(2.0/3.0) # it allows (1-xx)% overlap
RE_SUBSEQ_LEN_R=(1.0/3.0)  # recursively try lenc*xx subseq
RE_LMIN_R=2         # minimum length 2*lstep
RE_RETRY_RR=tl.YES  # if you want to re-try regimereader after RE/insertion
# --- RegimeReader
RR_TH_R=0.8    # th of regimereader e.g., th=0.8*|Xc| 
ERR_MIN=0.1    # minimum error value for everything
AVOIDINF_R=1.2 # _checkINF, ignore Ve if error(Vc-Xc) is too large
# --- nested structure
def _makeH_ma(lstep):
    H_ma=[2*lstep,  1]
    return H_ma
#------------------------------------------------#

#----------------------------#
#    model fitting           #
#----------------------------#
def run(X, lstep, MDBH, fn, dps_r, insertMDB=tl.YES):
    global UPDATE_MDB; UPDATE_MDB=insertMDB
    global DPS; DPS=max(1,int(lstep*dps_r));
    #--------------------------------#
    tl.comment("start RegimeCast") 
    #--------------------------------#
    _regime_cast(X, lstep, MDBH, fn)
    #--------------------------------#
    tl.comment("end RegimeCast")
    #--------------------------------#


#-----------------------------#
# RegimeReader
#-----------------------------#
# return estimated event Ve
def _regime_reader(Xc, lenc, lene, MDB):
    # (1) init params (r: # of regimes in MDB, d: dimension)
    r = len(MDB); d=np.size(Xc,1)
    Wt=np.zeros(r); err_c=tl.INF; Ve=np.zeros((lene,d))
    # if there is no regime in MDB, return 
    if(r==0): return (Wt, err_c, Ve, MDB)
    # (2) estimate opt event for each regime in MDB
    (MDB, Ve_r, err_r)=_estimate_optRegimes(Xc, lene, MDB)
    Vc_r=Ve_r[:,0:lenc,:] # Vc_r[r]: current estimated events for each regime 
    # (3) compute regime-activity (strengthi/weight of each regime) 
    Wt=_decompWt(Xc, Vc_r, r, d, lenc) 
    # (4) create estimated event 
    Ve=_mix_Ve_r(Ve_r, Ve, r, Wt)
    # (5) compute error (Xc-Vc) ... Vc=Ve[0:lenc]
    err_c=tl.RMSE(Xc,Ve[0:lenc,:]); err_org=tl.RMSE(Xc,[])
    # (6) if bad-fit, use current/latest event Xc[tm_ed]  
    if(err_c>err_org*RR_TH_R and err_c>ERR_MIN ):   
        Ve[0:lene,:]= Xc[lenc-1,:]; err_c=tl.RMSE(Xc,Ve[0:lenc,:]); 
    return (Wt, err_c, Ve, MDB) 
#-----------------------------#


#-----------------------------#
# RegimeEstimator 
#-----------------------------#
# estimate new regime, and insert it into MDB
def _regime_estimator(Xc, MDB, fn, level, ma, lstep):
    # (0) if cannot get new regime, return org MDB
    if(UPDATE_MDB==tl.NO or len(Xc)<lstep*RE_LMIN_R or len(MDB)>=MAXR): return MDB
    tl.msg("    |----RE: MDB(c=%d) n=%d (level:%d)"%(len(MDB), len(Xc), level))
    # (1) create new model mds 
    mds=NLDS(num_works=1, lam_list=[1.0, 1e-1, 1e-2], init_cov_list=[1.0], tol=0.01,
        fn="%s_mdb/H%d_r%d"%(fn,level, len(MDB)), ma=ma)
    # (2) estimate model params
    mds=mds.fit(Xc, fit_type='Latent')
    # if(MSAVE): mds.plot(mds.fn)
    # (3-a) generate and compute |Xc-Vc|
    (Sta, Obs)=mds.gen(mds.n); err=tl.RMSE(Xc,Obs)
    # (3-b) compute original |Xc|
    err_org=tl.RMSE(Xc,[])
    # (4-a) if it cannot find good-fit, then re-insert subsequence Xc*RE_SUBSEQ_LEN_R 
    if( err>ERR_MIN and err>err_org*RE_TH_R ): 
        if(DBG): tl.msg("    ........ bad-fit:  |Xc|:%f, |Xc-Vc|:%f (err_r:%f>th=%f)"%(err_org, err, err/err_org,RE_TH_R))
        # split r%-subseq and try RegEst again (delete (RE_SUBSEQ_LEN_R)-head)
        n=len(Xc); ed=n-1; st=int(n*RE_SUBSEQ_LEN_R)-1
        return _regime_estimator(Xc[st:ed,:], MDB, fn, level, ma, lstep)
    # (4-b) if it finds good-fit, then insert it into MDB
    else:
        tl.msg("    ........ insert Xc(n=%d) -> MDB(level=%d, c=%d)"%(len(Xc), level, len(MDB)))
        MDB.append(mds)
    tl.msg("    |----RE : (END) ")
    return MDB
#-----------------------------#

#-----------------------#
# RegimeCast (main func)
#-----------------------#
# given: 
#  Xt (n x 1)  : time list
#  X  (n x d)  : event stream 
#  lstep       : ls-steps-ahead forecasting
#  fn          : output file name
#-----------------------#
def _regime_cast(X, lstep, MDBH, fn):
    
    #------------------------------------------------------------#
    # set windows
    #------------------------------------------------------------#
    lenp=int(np.ceil(lstep*LP_R)) 
    lenc=int(lstep*RR_R)    # lenc: lc
    lens=lstep              # lens: ls
    lene=lenc+lens+lenp     # lene: lc+ls+lp 
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    # create hierarchical modelDB (MDBH) / event stream (XH)
    #------------------------------------------------------------#
    (MDBH, H_ma, HEIGHT) = _create_MDB(MDBH, lstep)
    (XH) = _create_XH(X, H_ma, HEIGHT)
    #------------------------------------------------------------#
    # data (X) size
    (n, d) = np.shape(X)
    # regime-shift-dynamics (i.e., w(t))
    RSH=np.zeros((HEIGHT,n,MAXR))
    # time & errors 
    ERR_full=np.zeros((n,2))  # estimated/forecasted event error
    TIME_full=np.zeros((n,2)) # speed 
    # estimated events (full/stream)
    Ve_full  =  np.nan*np.zeros((n,d))        # full
    VeH_full =  np.nan*np.zeros((HEIGHT,n,d)) # each level 
    ## forecasted events(full/stream)
    Vf_full  = np.nan*np.zeros((n,d))         # full
    VfH_full = np.nan*np.zeros((HEIGHT,n,d))  # each level
    # snapshots
    SS_Ticks=[]; SS_Ve=[]; 
    #------------------------------------------------------------#
    re_edH=(-1)*np.ones(HEIGHT) # for RegimeEstimator (latest insertion point for each level)
    #------------------------------------------------------------#
    

    #------------------------------------------------------------#
    # Start RegimeCast
    #------------------------------------------------------------#
    tf_ed=1 # start 
    while(True): 
        #------------------------------------------------------------#
        # set time ticks (tm_st, tm_ed, tf_st, tf_ed) 
        #------------------------------------------------------------#
        tc=tf_ed-lens   # current time tick (i.e., tc=tm_ed)
        tm_st=tc-lenc      # tm
        tm_ed=tc           # tc
        tf_st=tc+lens      # ts
        tf_ed=tc+lens+lenp # te
        if(tm_st<0): tf_ed=lene; continue; 
        #------------------------------------------------------------#
        tl.comment("RC: tc = %d"%(tm_ed))
        tl.msg("(tm, tc, ts, te) = (%d, %d, %d, %d)"%(tm_st, tm_ed, tf_st, tf_ed))
        #------------------------------------------------------------#

        #------------------------------------------------------------#
        # initialize modelDB (if MDBH is null) - RegimeEstimator 
        #------------------------------------------------------------#
        for i in range(0,HEIGHT):
            if(len(MDBH[i])==0):
                re_st=max(0,tm_ed-lene); re_ed=tm_ed; # insert X[re_st:re_ed] into modelDB
                MDBH[i]=_regime_estimator(XH[i,re_st:re_ed,:], MDBH[i], fn, i, H_ma[i], lstep)
                re_edH[i]=re_ed
        #------------------------------------------------------------#

        #===================#
        tic = time.process_time()
        #===================#
        re_time_i=0; # time-regime_est
        #===================#
        #------------------------------------------------------------#
        # nested structure 
        #------------------------------------------------------------#
        for i in range(0,HEIGHT):
            
            #------------------------------------------------------------#
            # create Xci, i.e., current window at level i
            #------------------------------------------------------------#
            Xci=XH[i,tm_st:tm_ed,:] 
            erri_org=tl.RMSE(Xci,[])
            
            #------------------------------------------------------------#
            # RegimeReader
            #------------------------------------------------------------#
            (Wti_rr, erri_rr, Vei_rr, MDBH[i])=_regime_reader(Xci, lenc, lene, MDBH[i])
            RSH[i,tf_st:tf_ed,0:len(MDBH[i])] = Wti_rr
            tl.msg("|----RR-Level-%d (%d) err(Xc-Vc): %.4f"%(i, H_ma[i], erri_rr))
            #------------------------------------------------------------#
            
            #------------------------------------------------------------#
            # if it cannot find good regime in MDB
            if(erri_rr>erri_org*RE_TH_R and erri_rr>ERR_MIN): 
                #===================#
                tic_i = time.process_time()
                #===================#
                #------------------------------------------------------------#
                # RegimeEstimator 
                #------------------------------------------------------------#
                re_st=max(0,tm_ed-lene); re_ed=tm_ed; # insert X[re_st:re_ed] into modelDB
                re_overlap=lene*RE_OVERLAP_LEN_R; diff=re_st-(re_edH[i]-re_overlap)
                if(re_edH[i]==-1 or diff>0): 
                    # if not overlapped window, then, insert it into MDB
                    MDBH[i]=_regime_estimator(XH[i,re_st:re_ed,:], MDBH[i], fn, i, H_ma[i], lstep)
                    re_edH[i]=re_ed # update latest insertion point
                    #------------------------------------------------------------#
                    # RegimeReader -retry (optional)
                    #------------------------------------------------------------#
                    if(RE_RETRY_RR):
                        (Wti_rr, erri_rr, Vei_rr, MDBH[i])=_regime_reader(Xci, lenc, lene, MDBH[i])
                        RSH[i,tf_st:tf_ed,0:len(MDBH[i])]=Wti_rr
                        tl.msg("    |----RR(retry)-Level-%d (%d) err(Xc-Vc): %.4f"%(i, H_ma[i], erri_rr))
                    #------------------------------------------------------------#
                #------------------------------------------------------------#
                #===================#
                toc_i = time.process_time(); re_time_i+= toc_i-tic_i;
                #===================#

            #------------------------------------------------------------#
            # update params at level i
            #------------------------------------------------------------#
            # model-fit
            VeH_full[i,tm_st:tf_ed,:]=Vei_rr[0:lene,:]
            # forecasted-seq
            VfH_full[i,tf_st:tf_ed,:]=Vei_rr[(lene-lenp):lene,:]
            #------------------------------------------------------------#
        
        #------------------------------------------------#
        # compute global/smoothed events  
        #------------------------------------------------#
        Ve_full[tm_st:tf_ed,:]=0; 
        Vf_full[tf_st:tf_ed,:]=0;
        for i in range(0,HEIGHT):
            my_ma=max(lenp,H_ma[i])
            my_tm_st=max(0,tm_st-my_ma)
            my_tf_st=max(0,tf_st-my_ma)
            VeH_tmp  = tl.smoothMAa(VeH_full[i, my_tm_st:tf_ed,:], my_ma)
            VfH_tmp  = tl.smoothMAa(VfH_full[i, my_tf_st:tf_ed,:], my_ma)
            Ve_full[tm_st:tf_ed,:] += VeH_tmp[len(VeH_tmp)-1-(tf_ed-tm_st):len(VeH_tmp)-1,:]
            Vf_full[tf_st:tf_ed,:] += VfH_tmp[len(VfH_tmp)-1-(tf_ed-tf_st):len(VfH_tmp)-1,:]
        #------------------------------------------------#
        #===================#
        toc = time.process_time(); fittime= toc-tic
        #===================#

        #------------------------------------------------#
        # time/error 
        #------------------------------------------------#
        # compute errors
        err_Vc_rr=tl.RMSE(X[tm_st:tm_ed,:],Ve_full[tm_st:tm_ed,:])  # current error  |Xc-Vc|
        err_Vf_rr=tl.RMSE(X[tf_st:tf_ed,:],Vf_full[tf_st:tf_ed,:])  # forecast error |Vf-Vf|
        # model error
        ERR_full[tf_st:tf_ed,:]=[err_Vc_rr, err_Vf_rr] # estimated errors 
        # times (regime-cast/regime-est) 
        TIME_full[tf_st:tf_ed,:]=[fittime, re_time_i]
        #------------------------------------------------#
        # update snapshot
        SS_Ve.append(tl.dcopy(Ve_full[tm_st:tf_ed]))        # estimated events 
        SS_Ticks.append([tm_st, tm_ed, tf_st, tf_ed])       # time-ticks
        #------------------------------------------------#
        tl.msg("........ time: %.4f (regimeRead: %.4f + regimeEst:%.4f)"%(fittime, (fittime-re_time_i), re_time_i))
        tl.msg("........ error: current window |Xc-Vc|=%.4f, forecast window |Xf-Vf|=%.4f"%(err_Vc_rr, err_Vf_rr))
        #------------------------------------------------#

        #------------------------------------------------#
        # save params
        #------------------------------------------------#
        finalSave=tl.NO
        if(tf_ed+lenp>=n): finalSave=tl.YES
        if(TSAVE or finalSave):
             # save snaps
            snaps={'Ve':np.asarray(SS_Ve), 'Ticks': SS_Ticks}
            tl.save_mat(snaps, "%sSnaps"%(fn))         # snapshot (format: matlab)
            #tl.save_obj(snaps, "%sSnaps"%(fn))        # snapshot (format: python_object)
            tl.save_obj(MDBH, '%sMDBH'%(fn))            # modelDB
            tl.save_txt(TIME_full, '%sTIME_full'%(fn))  # speed
            tl.save_txt(ERR_full,  '%sERR_full'%(fn))   # error 
            # data/params
            for i in range(0,HEIGHT): tl.save_txt(XH[i], '%sXH_%d'%(fn, i))
            for i in range(0,HEIGHT): tl.save_txt(RSH[i,:,0:len(MDBH[i])], '%sRSH_%d'%(fn, i))
            tl.save_txt(tl.smoothMAa(Vf_full,lenp), '%sVf_full'%(fn))
            for i in range(0,HEIGHT): tl.save_txt(tl.smoothMAa(VfH_full[i], max(lenp,H_ma[i])), '%sVfH_%d'%(fn, i))
        #------------------------------------------------#
            
        #------------------------------------------------#
        # terminate
        #------------------------------------------------#
        if(tf_ed+lenp>=n): return; 
        #------------------------------------------------#



#-----------------------#
# func-misc
#-----------------------#

#-----------------------------#
# for RegimeCast
#-----------------------------#
def _create_MDB(MDBH, lstep):
    tl.comment("create (hierarchical) MDB ")
    # MDBH: (hierarchical) regime list
    if(MDBH==[]):
        # create H_ma
        H_ma=_makeH_ma(lstep)
        HEIGHT=len(H_ma)
        # model query set
        MDBH=[] 
        for i in range(0,HEIGHT): MDBH.append([])
        # multi-scale smoothing
    else:
        H_ma=[]
        HEIGHT=len(MDBH)
        for i in range(0,HEIGHT): H_ma.append(MDBH[i][0].ma)
    return (MDBH, H_ma, HEIGHT)
#-----------------------------#
# create hierarchical event stream X
def _create_XH(X, H_ma, HEIGHT):
    XH=[]; X_tmp=tl.dcopy(X)
    # create smooth X
    for i in range(0,HEIGHT):
        h=H_ma[i]
        X_h=tl.smoothMA(X_tmp, h)
        XH.append(X_h)
        X_tmp = X_tmp-X_h
    XH=np.asarray(XH)
    return XH
#-----------------------------#




#-----------------------------#
# for RegimeReader
#-----------------------------#

#------------------------------------------------#
# if Obs is too far from Xorg, then return True
def _checkINF(Obs, Xorg):
    th=AVOIDINF_R
    mx=max(abs(Xorg.flatten()))
    if(sum(abs(Obs.flatten())>mx*th)>=1): return True
    else: return False
#------------------------------------------------#

#------------------------------------------------#
def _decompWt(Xc, Vc_r, r, d, lenc):
    #------------------------------------------------#
    # Given: 
    # Xc       (lenc x d)
    # Vc_r (r x lenc x d)
    # r: # of regimes
    # d: dimension
    # lenc: length of current window Xc
    #------------------------------------------------#
    # Out: 
    # Wt = {w[i]} = argmin ||Xc - sum(w[i]*Vc_r[i])|| (i=1,...r)
    #------------------------------------------------#
    # (1) find w[i] ... argmin(sum_{i}^{r} || Xc - w[i]*Vc_r[i] ||
    # (1-i) create weighted vector Wv
    Wv=tl.func_Weight(lenc,W_RR)
    #------------------------------------------------#
    # (1-ii) create flat vectors (Xc_flat/Vc_flat)
    Vc_r_flat=np.zeros((lenc*d, r)) # Vc_r (flatten) 
    Xc_flat=np.zeros((lenc*d, 1))   # Xc   (flatten)
    # (a) Vc_r_flat
    for i in range(0,r):
        for j in range(0,lenc):
            for jj in range(0,d):
                Vc_r_flat[j*d+jj][i]=Vc_r[i][j][jj]*Wv[j]
    # (b) Xc_flat
    for j in range(0,lenc):
        for jj in range(0,d):
            Xc_flat[j*d+jj][0]=Xc[j][jj]*Wv[j]
    #------------------------------------------------#
    # (2) find Wt={w[i]} (i=1,...r) Wt= argmin(||Xc - sum_{i}^{r} w[i]*Vc_r[i] ||
    #------------------------------------------------#
    Wt=la.lstsq(Vc_r_flat,Xc_flat, rcond=None)[0].T[0]
    # (a) remove/ignore small w[i]
    Wt[Wt<0.1]=0.0; Wt[Wt>1.0]=1.0; 
    #---------------------#
    # (b) re-try/find Wt
    Wt[Wt>0]=1.0;
    for i in range(0,r):
        Vc_r_flat[:,i]*=Wt[i]
    # (c) Wt=argmin||Xc-sum(Vc_r[i])||
    Wt=la.lstsq(Vc_r_flat,Xc_flat, rcond=None)[0].T[0]
    ## remove small Wt[i]
    #Wt[Wt<0.1]=0.0; Wt[Wt>1.0]=1.0; #notfin
    Wt[Wt<0.0]=0.0; Wt[Wt>1.0]=1.0; #notfin
    return Wt
#------------------------------------------------#

#------------------------------------------------#
def _mix_Ve_r(Ve_r, Ve, r, Wt):
    # compute mixed estimated events Ve = Wt*Ve_r 
    Ve[:]=0.0 
    for i in range(0,r):
        Ve+=Wt[i]*Ve_r[i]
    return Ve
#------------------------------------------------#

#------------------------------------------------#
def _estimate_optRegimes(Xc, lene, MDB):
    r = len(MDB); d=np.size(Xc,1)
    Ve_r=[]; err_r=[]
    err_org=tl.RMSE(Xc, [])
    # (1) for each regime in MDB, estimate Ve_r[i] (i=1,...r)
    for i in range(0,r):
        mds=MDB[i]; mds.data=Xc;  # regime mds in MDB
        mu0_org=tl.dcopy(mds.mu0)
        mds.fit_mu0(Xc, W_RR, DPS)     # update init mu0 (i.e., s(0)=mu0)
        (Sta,Obs)=mds.gen(lene)
        err=tl.RMSE(Xc, Obs[0:len(Xc)])  # rmse(Xc-Vc)
        # if err is too large, then, ignore this regime mds
        if(_checkINF(Obs,Xc) or (err>ERR_MIN and err>err_org*RR_TH_R)): 
            Obs[:]=0.0; err=tl.INF; mds.mu0=mu0_org;
        # update Ve_r[i], err_r[i] 
        Ve_r.append(Obs); err_r.append(err)
    Ve_r=np.asarray(Ve_r)
    return (MDB, Ve_r, err_r)
#------------------------------------------------#

if __name__ == "__main__":
    #--- arguments ---#
    parser = argparse.ArgumentParser()
    #-----------------------------------#
    parser.add_argument("-s",  "--seqfn", type=str, help="input seqs filename")
    parser.add_argument("-l",  "--lstep", type=int, help="l-step-ahead")
    parser.add_argument("-o",  "--outdir",type=str, help="output dir")
    #-----------------------------------#
    parser.add_argument("-q",  "--MDBfn", type=str, help="input modelDB filename (optional)")
    parser.add_argument("-i",  "--insert",type=str, help="insert/update MDB (optional)")
    parser.add_argument("-n",  "--n",     type=int, help="duration (optional)")
    parser.add_argument("-p",  "--dps", type=float, help="DPS-rate (default:0.1)")
    #-----------------------------------#
    args = parser.parse_args()
    if(len(sys.argv)<2):
        parser.print_help()
    #-----------------------------------#
    #--- check sequencefn ---#
    if(args.seqfn!=None):
        seqfn = args.seqfn
    else: parser.print_help()
    #--- check l-step-ahead ---#
    if(args.lstep!=None):
        lstep = args.lstep
    else: parser.print_help()
    #--- check output dir ---#
    if(args.outdir!=None): 
        outdir = args.outdir
    else: parser.print_help()
    #-----------------------------------#
    #--- check insert ---#
    if(args.insert=='no'): insert=tl.NO;
    else: insert = tl.YES;
    #--- check modelfn ---#
    if(args.MDBfn!=None):
        MDBfn = args.MDBfn
    else: MDBfn = ''
        #--- check n (duration) ---#
    if(args.n!=None): my_n=args.n
    else: my_n=tl.INF
    #--- check dps  ---#
    if(args.dps!=None): my_dps_r=args.dps
    else: my_dps_r=0.1
    #-----------------------------------#
    tl.comment("RegimeCast")
    #-----------------------------------#
    tl.comment('args/data/info')
    tl.msg("seqfn: [%s]"%seqfn)
    tl.msg("l-step-ahead: [%d]"%lstep)
    tl.msg("outdir:[%s]"%outdir)
    tl.msg("duration: [%s]"%my_n)
    tl.msg("queryfn (if any): [%s]"%MDBfn)
    tl.msg("update MDB: [%r] (1:yes,0:no)"%insert)
    tl.msg("dps_r: [%.1f]"%my_dps_r)
    tl.msg("dps: [%d]"%max(1,lstep*my_dps_r))
    #-----------------------------------#
    # create directory
    try:
        tl.mkdir("%s_mdb"%outdir)
    except:
        tl.error("cannot find: %s"%outdir)
    # load data (event stream)
    data=tl.loadsq(seqfn).T
    (n,d) = np.shape(data)
    # avoid zero/nan (kf) 
    data+=0.001*np.random.rand(n,d)
    #--- set data length (my_n) ---#
    if(my_n < n): 
        data=data[0:my_n,:]
        (n,d) = np.shape(data)
    data=tl.normalizeZ(data)
    tl.save_txt(data, '%sX'%(outdir))
    #--- load modelDB (if any) ---#
    MDBH=[];
    if(MDBfn!=''):
        try: MDBH=tl.load_obj("%s"%(MDBfn))
        except: tl.warning("cannot find: %s"%(MDBfn)); MQHs=[]
    
    #-----------------------------------#
    # start regimecast 
    #-----------------------------------#
    run(data, lstep, MDBH, outdir, my_dps_r, insert)
    #-----------------------------------#