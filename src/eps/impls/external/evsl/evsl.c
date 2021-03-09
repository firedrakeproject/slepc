/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to eigensolvers in EVSL.
*/

#include <slepc/private/epsimpl.h>    /*I "slepceps.h" I*/
#include <evsl.h>

typedef struct {
  PetscBool         initialized;
  Mat               A;           /* problem matrix */
  Vec               x,y;         /* auxiliary vectors */
  PetscReal         *sli;        /* slice bounds */
  PetscInt          nev;         /* approximate number of wanted eigenvalues in each slice */
  /* user parameters */
  PetscInt          nslices;     /* number of slices */
  PetscReal         lmin,lmax;   /* numerical range (min and max eigenvalue) */
  EPSEVSLDOSMethod  dos;         /* DOS method, either KPM or Lanczos */
  PetscInt          nvec;        /* number of sample vectors used for DOS */
  PetscInt          deg;         /* polynomial degree used for DOS (KPM only) */
  EPSEVSLKPMDamping damping;     /* type of damping used for DOS (KPM only) */
  PetscInt          steps;       /* number of Lanczos steps used for DOS (Lanczos only) */
  PetscInt          npoints;     /* number of sample points used for DOS (Lanczos only) */
} EPS_EVSL;

static void AMatvec_EVSL(double *xa,double *ya,void *data)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)data;
  Vec            x = ctx->x,y = ctx->y;
  Mat            A = ctx->A;

  PetscFunctionBegin;
  ierr = VecPlaceArray(x,(PetscScalar*)xa);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = VecPlaceArray(y,(PetscScalar*)ya);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = MatMult(A,x,y);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = VecResetArray(x);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = VecResetArray(y);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  PetscFunctionReturnVoid();
}

PetscErrorCode EPSSetUp_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;
  PetscMPIInt    size;
  PetscBool      isshift;
  PetscScalar    *vinit;
  PetscReal      *mu,ecount,xintv[4],*xdos,*ydos;
  Vec            v0;
  PetscRandom    rnd;

  PetscFunctionBegin;
  EPSCheckStandard(eps);
  EPSCheckHermitian(eps);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size);CHKERRMPI(ierr);
  if (size>1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This version of EVSL does not support MPI parallelism");
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (!isshift) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");

  if (!ctx->initialized) EVSLStart();

  /* get matrix and prepare auxiliary vectors */
  ierr = STGetMatrix(eps->st,0,&ctx->A);CHKERRQ(ierr);
  SetAMatvec(eps->n,&AMatvec_EVSL,(void*)ctx);
  if (!ctx->x) {
    ierr = MatCreateVecsEmpty(ctx->A,&ctx->x,&ctx->y);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->x);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->y);CHKERRQ(ierr);
  }
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);

  if (!eps->which) eps->which=EPS_ALL;
  if (eps->which!=EPS_ALL || eps->inta==eps->intb) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver requires setting an interval with EPSSetInterval()");

  /* estimate numerical range */
  if (ctx->lmin == PETSC_MIN_REAL || ctx->lmax == PETSC_MAX_REAL) {
    ierr = STMatCreateVecs(eps->st,&v0,NULL);CHKERRQ(ierr);
    ierr = BVGetRandomContext(eps->V,&rnd);CHKERRQ(ierr);
    ierr = VecSetRandom(v0,rnd);CHKERRQ(ierr);
    ierr = VecGetArray(v0,&vinit);CHKERRQ(ierr);
    ierr = LanTrbounds(50,200,eps->tol,vinit,1,&ctx->lmin,&ctx->lmax,NULL);CHKERRQ(ierr);
    ierr = VecRestoreArray(v0,&vinit);CHKERRQ(ierr);
    ierr = VecDestroy(&v0);CHKERRQ(ierr);
  }
  if (ctx->lmin > eps->inta || ctx->lmax < eps->intb) SETERRQ4(PetscObjectComm((PetscObject)eps),1,"The requested interval [%g,%g] must be contained in the numerical range [%g,%g]",(double)eps->inta,(double)eps->intb,(double)ctx->lmin,(double)ctx->lmax);
  xintv[0] = eps->inta;
  xintv[1] = eps->intb;
  xintv[2] = ctx->lmin;
  xintv[3] = ctx->lmax;

  /* estimate number of eigenvalues in the interval */
  if (ctx->dos == EPS_EVSL_DOS_KPM) {
    ierr = PetscMalloc1(ctx->deg+1,&mu);CHKERRQ(ierr);
    ierr = kpmdos(ctx->deg,(int)ctx->damping,ctx->nvec,xintv,mu,&ecount);CHKERRQ(ierr);
  } else if (ctx->dos == EPS_EVSL_DOS_LANCZOS) {
    ierr = PetscMalloc2(ctx->npoints,&xdos,ctx->npoints,&ydos);CHKERRQ(ierr);
    ierr = LanDos(ctx->nvec,PetscMin(ctx->steps,eps->n/2),ctx->npoints,xdos,ydos,&ecount,xintv);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid DOS method");

  ierr = PetscInfo1(eps,"Estimated eigenvalue count in the interval: %g\n",ecount);CHKERRQ(ierr);
  eps->ncv = (PetscInt)PetscCeilReal(1.5*ecount);

  /* slice the spectrum */
  ierr = PetscMalloc1(ctx->nslices+1,&ctx->sli);CHKERRQ(ierr);
  if (ctx->dos == EPS_EVSL_DOS_KPM) {
    ierr = spslicer(ctx->sli,mu,ctx->deg,xintv,ctx->nslices,10*(PetscInt)ecount);CHKERRQ(ierr);
    ierr = PetscFree(mu);CHKERRQ(ierr);
  } else if (ctx->dos == EPS_EVSL_DOS_LANCZOS) {
    spslicer2(xdos,ydos,ctx->nslices,ctx->npoints,ctx->sli);
    ierr = PetscFree2(xdos,ydos);CHKERRQ(ierr);
  }

  /* approximate number of eigenvalues wanted in each slice */
  ctx->nev = (PetscInt)(1.0 + ecount/(PetscReal)ctx->nslices) + 2;

  if (eps->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;
  PetscInt       i,k=0,sl,mlan,nevout,*ind;
  PetscReal      *res,xintv[4];
  PetscScalar    *lam,*Y,*vinit,*pV;
  PetscRandom    rnd;
  Vec            v0;
  polparams      pol;

  PetscFunctionBegin;
  ierr = BVCreateVec(eps->V,&v0);CHKERRQ(ierr);
  ierr = BVGetRandomContext(eps->V,&rnd);CHKERRQ(ierr);
  ierr = VecSetRandom(v0,rnd);CHKERRQ(ierr);
  ierr = VecGetArray(v0,&vinit);CHKERRQ(ierr);
  ierr = BVGetArray(eps->V,&pV);CHKERRQ(ierr);
  mlan = PetscMin(PetscMax(5*ctx->nev,300),eps->n);
  for (sl=0; sl<ctx->nslices; sl++) {
    xintv[0] = ctx->sli[sl];
    xintv[1] = ctx->sli[sl+1];
    xintv[2] = ctx->lmin;
    xintv[3] = ctx->lmax;
    ierr = PetscInfo3(eps,"Subinterval %D: [%.4e, %.4e]\n",sl+1,xintv[0],xintv[1]);CHKERRQ(ierr);
    set_pol_def(&pol);
    /*
    //-------------------- this is to show how you can reset some of the
    //                     parameters to determine the filter polynomial
    pol.damping = 2;
    //-------------------- use a stricter requirement for polynomial
    pol.thresh_int = 0.8;
    pol.thresh_ext = 0.2;
    pol.max_deg  = 3000;
    // pol.deg = 20 //<< this will force this exact degree . not recommended
    //                   it is better to change the values of the thresholds
    //                   pol.thresh_ext and plot.thresh_int
    //-------------------- Now determine polymomial to use
    */
    find_pol(xintv,&pol);
    ierr = PetscInfo4(eps,"Polynomial [type = %D], deg %D, bar %e gam %e\n",pol.type,pol.deg,pol.bar,pol.gam);CHKERRQ(ierr);
    ierr = ChebLanNr(xintv,mlan,eps->tol,vinit,&pol,&nevout,&lam,&Y,&res,NULL);CHKERRQ(ierr);
    if (k+nevout>eps->ncv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Too low estimation of eigenvalue count, try modifying the sampling parameters");
    free_pol(&pol);
    ierr = PetscInfo1(eps,"Computed %D eigenvalues\n",nevout);CHKERRQ(ierr);
    ierr = PetscMalloc1(nevout,&ind);CHKERRQ(ierr);
    sort_double(nevout,lam,ind);
    for (i=0;i<nevout;i++) {
      eps->eigr[i+k]   = lam[i];
      eps->errest[i+k] = res[ind[i]];
      ierr = PetscArraycpy(pV+(i+k)*eps->nloc,Y+ind[i]*eps->nloc,eps->nloc);CHKERRQ(ierr);
    }
    k += nevout;
    if (lam) { evsl_Free(lam); }
    if (Y) { evsl_Free_device(Y); }
    if (res) { evsl_Free(res); }
    ierr = PetscFree(ind);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v0,&vinit);CHKERRQ(ierr);
  ierr = VecDestroy(&v0);CHKERRQ(ierr);
  ierr = BVRestoreArray(eps->V,&pV);CHKERRQ(ierr);

  eps->nev    = k;
  eps->nconv  = k;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetRange_EVSL(EPS eps,PetscReal lmin,PetscReal lmax)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (lmin>lmax) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be lmin<lmax");
  if (ctx->lmin != lmin || ctx->lmax != lmax) {
    ctx->lmin  = lmin;
    ctx->lmax  = lmax;
    eps->state = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetRange - Defines the numerical range (or field of values) of the problem,
   that is, the interval containing all eigenvalues.

   Logically Collective on eps

   Input Parameters:
+  eps  - the eigensolver context
.  lmin - left end of the interval
-  lmax - right end of the interval

   Options Database Key:
.  -eps_evsl_range <a,b> - set [a,b] as the numerical range

   Notes:
   The filter will be most effective if the numerical range is tight, that is, lmin
   and lmax are good approximations to the leftmost and rightmost eigenvalues,
   respectively. If not set by the user, an approximation is computed internally.

   The wanted computational interval specified via EPSSetInterval() must be
   contained in the numerical range.

   Level: intermediate

.seealso: EPSEVSLGetRange(), EPSSetInterval()
@*/
PetscErrorCode EPSEVSLSetRange(EPS eps,PetscReal lmin,PetscReal lmax)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,lmin,2);
  PetscValidLogicalCollectiveReal(eps,lmax,3);
  ierr = PetscTryMethod(eps,"EPSEVSLSetRange_C",(EPS,PetscReal,PetscReal),(eps,lmin,lmax));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetRange_EVSL(EPS eps,PetscReal *lmin,PetscReal *lmax)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (lmin) *lmin = ctx->lmin;
  if (lmax) *lmax = ctx->lmax;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetRange - Gets the interval containing all eigenvalues.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameters:
+  lmin - left end of the interval
-  lmax - right end of the interval

   Level: intermediate

.seealso: EPSEVSLSetRange()
@*/
PetscErrorCode EPSEVSLGetRange(EPS eps,PetscReal *lmin,PetscReal *lmax)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSEVSLGetRange_C",(EPS,PetscReal*,PetscReal*),(eps,lmin,lmax));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetDOSParameters_EVSL(EPS eps,EPSEVSLDOSMethod dos,PetscInt nvec,PetscInt deg,EPSEVSLKPMDamping damping,PetscInt steps,PetscInt npoints)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ctx->dos = dos;
  if (nvec == PETSC_DECIDE || nvec == PETSC_DEFAULT) ctx->nvec = 80;
  else if (nvec<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The nvec argument must be > 0");
  else ctx->nvec = nvec;
  switch (dos) {
    case EPS_EVSL_DOS_KPM:
      if (deg == PETSC_DECIDE || deg == PETSC_DEFAULT) ctx->deg = 300;
      else if (deg<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The deg argument must be > 0");
      else ctx->deg = deg;
      ctx->damping = damping;
      break;
    case EPS_EVSL_DOS_LANCZOS:
      if (steps == PETSC_DECIDE || steps == PETSC_DEFAULT) ctx->steps = 40;
      else if (steps<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The steps argument must be > 0");
      else ctx->steps = steps;
      if (npoints == PETSC_DECIDE || npoints == PETSC_DEFAULT) ctx->npoints = 200;
      else if (npoints<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The npoints argument must be > 0");
      else ctx->npoints = npoints;
      break;
  }
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetDOSParameters - Defines the parameters used for computing the
   density of states (DOS) in the EVSL solver.

   Logically Collective on eps

   Input Parameters:
+  eps     - the eigensolver context
.  dos     - DOS method, either KPM or Lanczos
.  nvec    - number of sample vectors
.  deg     - polynomial degree (KPM only)
.  damping - type of damping (KPM only)
.  steps   - number of Lanczos steps (Lanczos only)
-  npoints - number of sample points (Lanczos only)

   Options Database Keys:
+  -eps_evsl_dos_method <dos> - set the DOS method, either kpm or lanczos
.  -eps_evsl_dos_nvec <n> - set the number of sample vectors
.  -eps_evsl_dos_degree <n> - set the polynomial degree
.  -eps_evsl_dos_damping <d> - set the type of damping
.  -eps_evsl_dos_steps <n> - set the number of Lanczos steps
-  -eps_evsl_dos_npoints <n> - set the number of sample points

   Notes:
   The density of states (or spectral density) can be approximated with two
   methods: kernel polynomial method (KPM) or Lanczos. Some parameters for
   these methods can be set by the user with this function, with some of
   them being relevant for one of the methods only.

   Level: intermediate

.seealso: EPSEVSLGetDOSParameters()
@*/
PetscErrorCode EPSEVSLSetDOSParameters(EPS eps,EPSEVSLDOSMethod dos,PetscInt nvec,PetscInt deg,EPSEVSLKPMDamping damping,PetscInt steps,PetscInt npoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,dos,2);
  PetscValidLogicalCollectiveInt(eps,nvec,3);
  PetscValidLogicalCollectiveInt(eps,deg,4);
  PetscValidLogicalCollectiveEnum(eps,damping,5);
  PetscValidLogicalCollectiveInt(eps,steps,6);
  PetscValidLogicalCollectiveInt(eps,npoints,7);
  ierr = PetscTryMethod(eps,"EPSEVSLSetDOSParameters_C",(EPS,EPSEVSLDOSMethod,PetscInt,PetscInt,EPSEVSLKPMDamping,PetscInt,PetscInt),(eps,dos,nvec,deg,damping,steps,npoints));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetDOSParameters_EVSL(EPS eps,EPSEVSLDOSMethod *dos,PetscInt *nvec,PetscInt *deg,EPSEVSLKPMDamping *damping,PetscInt *steps,PetscInt *npoints)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (dos)     *dos     = ctx->dos;
  if (nvec)    *nvec    = ctx->nvec;
  if (deg)     *deg     = ctx->deg;
  if (damping) *damping = ctx->damping;
  if (steps)   *steps   = ctx->steps;
  if (npoints) *npoints = ctx->npoints;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetDOSParameters - Gets the parameters used for computing the
   density of states (DOS) in the EVSL solver.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameters:
+  dos     - DOS method, either KPM or Lanczos
.  nvec    - number of sample vectors
.  deg     - polynomial degree (KPM only)
.  damping - type of damping (KPM only)
.  steps   - number of Lanczos steps (Lanczos only)
-  npoints - number of sample points (Lanczos only)

   Level: intermediate

.seealso: EPSEVSLSetDOSParameters()
@*/
PetscErrorCode EPSEVSLGetDOSParameters(EPS eps,EPSEVSLDOSMethod *dos,PetscInt *nvec,PetscInt *deg,EPSEVSLKPMDamping *damping,PetscInt *steps,PetscInt *npoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSEVSLGetDOSParameters_C",(EPS,EPSEVSLDOSMethod*,PetscInt*,PetscInt*,EPSEVSLKPMDamping*,PetscInt*,PetscInt*),(eps,dos,nvec,deg,damping,steps,npoints));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_EVSL(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  numerical range = [%g,%g]\n",(double)ctx->lmin,(double)ctx->lmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  computing DOS with %s: nvec=%D, ",EPSEVSLDOSMethods[ctx->dos],ctx->nvec);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    switch (ctx->dos) {
      case EPS_EVSL_DOS_KPM:
        ierr = PetscViewerASCIIPrintf(viewer,"degree=%D, damping=%s\n",ctx->deg,EPSEVSLKPMDampings[ctx->damping]);CHKERRQ(ierr);
        break;
      case EPS_EVSL_DOS_LANCZOS:
        ierr = PetscViewerASCIIPrintf(viewer,"steps=%D, npoints=%D\n",ctx->steps,ctx->npoints);CHKERRQ(ierr);
        break;
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_EVSL(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode    ierr;
  PetscReal         array[2]={0,0};
  PetscInt          k,i1,i2,i3,i4;
  PetscBool         flg,flg1;
  EPSEVSLDOSMethod  dos;
  EPSEVSLKPMDamping damping;
  EPS_EVSL          *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS EVSL Options");CHKERRQ(ierr);

    k = 2;
    ierr = PetscOptionsRealArray("-eps_evsl_range","Interval containing all eigenvalues (two real values separated with a comma without spaces)","EPSEVSLSetRange",array,&k,&flg);CHKERRQ(ierr);
    if (flg) {
      if (k<2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_SIZ,"Must pass two values in -eps_evsl_range (comma-separated without spaces)");
      ierr = EPSEVSLSetRange(eps,array[0],array[1]);CHKERRQ(ierr);
    }

    ierr = EPSEVSLGetDOSParameters(eps,&dos,&i1,&i2,&damping,&i3,&i4);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-eps_evsl_dos_method","Method to compute the DOS","EPSEVSLSetDOSParameters",EPSEVSLDOSMethods,(PetscEnum)ctx->dos,(PetscEnum*)&dos,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_evsl_dos_nvec","Number of sample vectors for DOS","EPSEVSLSetDOSParameters",i1,&i1,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsInt("-eps_evsl_dos_degree","Polynomial degree used for DOS","EPSEVSLSetDOSParameters",i2,&i2,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsEnum("-eps_evsl_dos_damping","Type of damping used for DOS","EPSEVSLSetDOSParameters",EPSEVSLKPMDampings,(PetscEnum)ctx->damping,(PetscEnum*)&damping,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsInt("-eps_evsl_dos_steps","Number of Lanczos steps in DOS","EPSEVSLSetDOSParameters",i3,&i3,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsInt("-eps_evsl_dos_npoints","Number of sample points used for DOS","EPSEVSLSetDOSParameters",i4,&i4,&flg1);CHKERRQ(ierr);
    if (flg || flg1) { ierr = EPSEVSLSetDOSParameters(eps,dos,i1,i2,damping,i3,i4);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (ctx->initialized) EVSLFinish();
  ierr = PetscFree(ctx->sli);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetRange_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetRange_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetDOSParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetDOSParameters_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->x);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_EVSL(EPS eps)
{
  EPS_EVSL       *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  ctx->nslices = 1;
  ctx->lmin    = PETSC_MIN_REAL;
  ctx->lmax    = PETSC_MAX_REAL;
  ctx->dos     = EPS_EVSL_DOS_KPM;
  ctx->nvec    = 80;
  ctx->deg     = 300;
  ctx->damping = EPS_EVSL_KPM_SIGMA;
  ctx->steps   = 40;
  ctx->npoints = 200;

  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_EVSL;
  eps->ops->setup          = EPSSetUp_EVSL;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->setfromoptions = EPSSetFromOptions_EVSL;
  eps->ops->destroy        = EPSDestroy_EVSL;
  eps->ops->reset          = EPSReset_EVSL;
  eps->ops->view           = EPSView_EVSL;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetRange_C",EPSEVSLSetRange_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetRange_C",EPSEVSLGetRange_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetDOSParameters_C",EPSEVSLSetDOSParameters_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetDOSParameters_C",EPSEVSLGetDOSParameters_EVSL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

