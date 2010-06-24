/*
  SLEPc eigensolver: "davidson"

  Method: General Davidson Method

  References:
    - Ernest R. Davidson. Super-matrix methods. Computer Physics Communications,
      53:49–60, May 1989.
*/

#include "private/epsimpl.h"
#include "private/stimpl.h"
#include "davidson.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_DAVIDSON"
PetscErrorCode EPSCreate_DAVIDSON(EPS eps) {
  PetscErrorCode  ierr;
  EPS_DAVIDSON    *data;

  PetscFunctionBegin;

  STSetType(eps->OP, STSHELL);
  STShellSetApply(eps->OP, SLEPcNotImplemented);
  STShellSetApplyTranspose(eps->OP, SLEPcNotImplemented);

  eps->OP->ops->getbilinearform  = STGetBilinearForm_Default;
  eps->ops->solve                = EPSSolve_DAVIDSON;
  eps->ops->setup                = EPSSetUp_DAVIDSON;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_QZ;

  ierr = PetscMalloc(sizeof(EPS_DAVIDSON), &data); CHKERRQ(ierr);
  eps->data = data;

  /* Set default values */
  ierr = EPSDAVIDSONSetKrylovStart_DAVIDSON(eps, PETSC_FALSE); CHKERRQ(ierr);
  ierr = EPSDAVIDSONSetBlockSize_DAVIDSON(eps, 1); CHKERRQ(ierr);
  ierr = EPSDAVIDSONSetRestart_DAVIDSON(eps, 6, 0); CHKERRQ(ierr);
  ierr = EPSDAVIDSONSetInitialSize_DAVIDSON(eps, 5); CHKERRQ(ierr);

  dvd_prof_init(&data->prof);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_DAVIDSON"
PetscErrorCode EPSSetUp_DAVIDSON(EPS eps) {
  PetscErrorCode  ierr;
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;
  dvdDashboard    *dvd = &data->ddb;
  dvdBlackboard   b;
  PetscInt        i,nvecs,nscalars,min_size_V,plusk,bs;
  Mat             A,B;
  KSP             ksp;
  PC              pc, pc2;
  PetscTruth      t,ipB;
  HarmType_t      harm;
  InitType_t      init;

  PetscFunctionBegin;

  /* Setup EPS options and get the problem specification */
  if(eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  } else if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
  else eps->ncv = PetscMin(eps->n,eps->nev+500);
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY))
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSDAVIDSONGetBlockSize_DAVIDSON(eps, &bs); CHKERRQ(ierr);
  if (bs <= 0) bs = 1;
  if (!(eps->nev + bs < eps->ncv))
    SETERRQ(1, "The ncv has to be greater than nev plus blocksize!");

  ierr = EPSDAVIDSONGetRestart_DAVIDSON(eps, &min_size_V, &plusk);
  CHKERRQ(ierr);
  if (min_size_V == 0) min_size_V = bs;
  if (!(min_size_V <= eps->ncv))
    SETERRQ(1, "The value of eps_davidsones_minv must be less than ncv!");
  if(eps->nini<=0) eps->nini = 5;

  ierr = STGetOperators(eps->OP, &A, &B); CHKERRQ(ierr);
  ierr = STGetKSP(eps->OP, &ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc, PCNONE, &t); CHKERRQ(ierr);
  if (t == PETSC_TRUE) {
    pc = 0;
  } else {
    PetscObjectReference((PetscObject)pc);
    ierr = PCCreate(((PetscObject)eps)->comm, &pc2); CHKERRQ(ierr);
    ierr = PCSetType(pc2, PCNONE); CHKERRQ(ierr);
    ierr = KSPSetPC(ksp, pc2); CHKERRQ(ierr);
    ierr = PCDestroy(pc2); CHKERRQ(ierr);
  }

  /* Setup problem specification in dvd */
  ierr = PetscMemzero(dvd, sizeof(dvdDashboard)); CHKERRQ(ierr);
  dvd->A = A; dvd->B = (eps->isgeneralized==PETSC_TRUE) ? B : PETSC_NULL;
  dvd->sA = DVD_MAT_IMPLICIT |
            (eps->ishermitian == PETSC_TRUE ? DVD_MAT_HERMITIAN : 0) |
	    (((eps->ispositive == PETSC_TRUE) &&
	      (eps->isgeneralized == PETSC_FALSE)) ? DVD_MAT_POS_DEF : 0);
  /* Asume -eps_hermitian means hermitian-definite in generalized problems */
  if ((eps->ispositive == PETSC_FALSE) &&
      (eps->isgeneralized == PETSC_FALSE) &&
      (eps->ishermitian == PETSC_TRUE)) eps->ispositive = PETSC_TRUE;
  if (eps->isgeneralized == PETSC_FALSE)
    dvd->sB = DVD_MAT_IMPLICIT | DVD_MAT_HERMITIAN | DVD_MAT_IDENTITY |
              DVD_MAT_UNITARY | DVD_MAT_POS_DEF;
  else 
    dvd->sB = DVD_MAT_IMPLICIT |
              (eps->ishermitian == PETSC_TRUE ? DVD_MAT_HERMITIAN : 0) |
              (eps->ispositive == PETSC_TRUE ? DVD_MAT_POS_DEF : 0);
  ipB = DVD_IS(dvd->sB, DVD_MAT_POS_DEF)?PETSC_TRUE:PETSC_FALSE;
  dvd->sEP = ((eps->isgeneralized == PETSC_FALSE) ||
              ( (eps->isgeneralized == PETSC_TRUE) &&
                (ipB == PETSC_TRUE)             ) ? DVD_EP_STD : 0) |
	     (eps->ispositive == PETSC_TRUE ? DVD_EP_HERMITIAN : 0);
  dvd->nev = eps->nev;
  dvd->which = eps->which;
  switch(eps->which) {
  case EPS_TARGET_MAGNITUDE:
  case EPS_TARGET_REAL:
  case EPS_TARGET_IMAGINARY:
    dvd->withTarget = PETSC_TRUE;
    dvd->target[0] = eps->target; dvd->target[1] = 1.0;
    break;

  case EPS_LARGEST_MAGNITUDE:
  case EPS_LARGEST_REAL:
  case EPS_LARGEST_IMAGINARY: //TODO: think about this case
  default:
    dvd->withTarget = PETSC_TRUE;
    dvd->target[0] = 1.0; dvd->target[1] = 0.0;
    break;
 
  case EPS_SMALLEST_MAGNITUDE:
  case EPS_SMALLEST_REAL:
  case EPS_SMALLEST_IMAGINARY: //TODO: think about this case
    dvd->withTarget = PETSC_TRUE;
    dvd->target[0] = 0.0; dvd->target[1] = 1.0;
  }
  dvd->tol = eps->tol;
  dvd->eps = eps;

  /* Setup the extraction technique */
  switch(eps->extraction) {
  case 0:
  case EPS_RITZ:              harm = DVD_HARM_NONE; break;
  case EPS_HARMONIC:          harm = DVD_HARM_RR; break;
  case EPS_HARMONIC_RELATIVE: harm = DVD_HARM_RRR; break;
  case EPS_HARMONIC_RIGHT:    harm = DVD_HARM_REIGS; break;
  case EPS_HARMONIC_LARGEST:  harm = DVD_HARM_LEIGS; break;
  default: SETERRQ(PETSC_ERR_SUP,"Unsupported extraction type");
  }

  /* Setup the type of starting subspace */
  ierr = EPSDAVIDSONGetKrylovStart_DAVIDSON(eps, &t); CHKERRQ(ierr);
  init = t==PETSC_FALSE ? DVD_INITV_CLASSIC : DVD_INITV_KRYLOV;

  /* Setup IP */
  if ((ipB == PETSC_TRUE) && (dvd->B)) {
    ierr = IPSetBilinearForm(eps->ip, dvd->B, IP_INNER_HERMITIAN); CHKERRQ(ierr);
  } else {
    ierr = IPSetBilinearForm(eps->ip, 0, IP_INNER_HERMITIAN); CHKERRQ(ierr);
  }

  /* Setup the random seed */
  ierr = PetscRandomCreate(((PetscObject)eps)->comm, &dvd->rand); CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(dvd->rand); CHKERRQ(ierr);

  /* Orthonormalize the DS */
  ierr = dvd_orthV(eps->ip, PETSC_NULL, 0, PETSC_NULL, 0, eps->DS, 0, eps->nds,
                   PETSC_NULL, 0, dvd->rand); CHKERRQ(ierr);

  /* The Davidson solver computes the residual vector and its norm, so
     EPSResidualConverged is replaced by EPSDefaultConverged */
  if (eps->conv_func == EPSResidualConverged)
    eps->conv_func = EPSDefaultConverged;

  /* Preconfigure dvd */
  ierr = dvd_schm_basic_preconf(dvd, &b, eps->ncv, min_size_V, bs,
                                eps->nini, eps->IS,
                                eps->IS?eps->nini:0,
                                plusk, pc, harm,
                                PETSC_NULL, init);
  CHKERRQ(ierr);

  /* Reserve memory */
  nvecs = b.max_size_auxV + b.own_vecs;
  nscalars = b.own_scalars + b.max_size_auxS;
  ierr = PetscMalloc((nvecs*eps->nloc+nscalars)*sizeof(PetscScalar), &data->wS);
  CHKERRQ(ierr);
  ierr = PetscMalloc(nvecs*sizeof(Vec), &data->wV); CHKERRQ(ierr);
  for (i=0; i<nvecs; i++) {
    ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm, eps->nloc, PETSC_DECIDE,
                                 data->wS+i*eps->nloc, &data->wV[i]);
    CHKERRQ(ierr);
  }
  b.free_vecs = data->wV;
  b.free_scalars = data->wS + nvecs*eps->nloc;
  dvd->auxV = data->wV + b.own_vecs;
  dvd->auxS = b.free_scalars + b.own_scalars;
  dvd->size_auxV = b.max_size_auxV;
  dvd->size_auxS = b.max_size_auxS;

  /* Configure dvd for a basic GD */
  ierr = dvd_schm_basic_conf(dvd, &b, eps->ncv, min_size_V, bs,
                             eps->nini, eps->IS,
                             eps->IS?eps->nini:0, plusk, pc,
                             eps->ip, harm, dvd->withTarget,
                             eps->target, ksp,
                             1.0, init);
  CHKERRQ(ierr);

  /* Associate the eigenvalues to the EPS */
  eps->eigr = dvd->eigr;
  eps->eigi = dvd->eigi;
  eps->errest = dvd->errest;
  eps->V = dvd->V;

  /* Prepare the profiler */
  dvd_profiler(dvd, data->prof);
   
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_DAVIDSON"
PetscErrorCode EPSSolve_DAVIDSON(EPS eps) {
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;
  dvdDashboard    *d = &data->ddb;
  PetscInt        r;

  PetscFunctionBegin;

  /* Call the starting routines */
  DVD_FL_CALL(d->startList, d);

  for(eps->its=0; eps->its < eps->max_it; eps->its++) {
    /* Initialize V, if it is needed */
    if (d->size_V == 0) r = d->initV(d);

    /* Find the best approximated eigenpairs in V, X */
    r = d->calcPairs(d);

    /* Expand the subspace */
    r = d->updateV(d);

    /* Monitor progress */
    eps->nconv = d->nconv;
//    EPSMonitor(eps, eps->its+1, eps->nconv, eps->eigr, eps->eigi, eps->errest, eps->nconv+PetscMax(d->size_H,5)); 
    EPSMonitor(eps, eps->its+1, eps->nconv, eps->eigr+eps->nconv, eps->eigi, eps->errest+eps->nconv, PetscMin(d->size_H,5));

    /* Test for convergence */
    if (eps->nconv >= eps->nev) break;
  }

  /* Call the ending routines */
  DVD_FL_CALL(d->endList, d);

  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SLEPcNotImplemented"
PetscErrorCode SLEPcNotImplemented() {
  SETERRQ(1, "Not call this function!");
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONSetKrylovStart_DAVIDSON"
PetscErrorCode EPSDAVIDSONSetKrylovStart_DAVIDSON(EPS eps,PetscTruth krylovstart)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  data->krylovstart = krylovstart;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONGetKrylovStart_DAVIDSON"
PetscErrorCode EPSDAVIDSONGetKrylovStart_DAVIDSON(EPS eps,PetscTruth *krylovstart)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  *krylovstart = data->krylovstart;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONSetBlockSize_DAVIDSON"
PetscErrorCode EPSDAVIDSONSetBlockSize_DAVIDSON(EPS eps,PetscInt blocksize)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if(blocksize == PETSC_DEFAULT || blocksize == PETSC_DECIDE) blocksize = 1;
  if(blocksize <= 0)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid blocksize value");
  data->blocksize = blocksize;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONGetBlockSize_DAVIDSON"
PetscErrorCode EPSDAVIDSONGetBlockSize_DAVIDSON(EPS eps,PetscInt *blocksize)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  *blocksize = data->blocksize;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONSetRestart_DAVIDSON"
PetscErrorCode EPSDAVIDSONSetRestart_DAVIDSON(EPS eps,PetscInt minv,PetscInt plusk)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if(minv == PETSC_DEFAULT || minv == PETSC_DECIDE) minv = 5;
  if(minv <= 0)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid minv value");
  if(plusk == PETSC_DEFAULT || plusk == PETSC_DECIDE) plusk = 5;
  if(plusk < 0)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid plusk value");
  data->minv = minv;
  data->plusk = plusk;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONGetRestart_DAVIDSON"
PetscErrorCode EPSDAVIDSONGetRestart_DAVIDSON(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  *minv = data->minv;
  *plusk = data->plusk;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONGetInitialSize_DAVIDSON"
PetscErrorCode EPSDAVIDSONGetInitialSize_DAVIDSON(EPS eps,PetscInt *initialsize)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  *initialsize = data->initialsize;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDAVIDSONSetInitialSize_DAVIDSON"
PetscErrorCode EPSDAVIDSONSetInitialSize_DAVIDSON(EPS eps,PetscInt initialsize)
{
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if(initialsize == PETSC_DEFAULT || initialsize == PETSC_DECIDE) initialsize = 5;
  if(initialsize <= 0)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid blocksize value");
  data->initialsize = initialsize;

  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_QZ"
/*
  EPSComputeVectors_QZ - Compute eigenvectors from the vectors
  provided by the eigensolver. This version is intended for solvers 
  that provide Schur vectors from the QZ decompositon. Given the partial
  Schur decomposition OP*V=V*T, the following steps are performed:
      1) compute eigenvectors of (S,T): S*Z=T*Z*D
      2) compute eigenvectors of OP: X=V*Z
  If left eigenvectors are required then also do Z'*Tl=D*Z', Y=W*Z
 */
PetscErrorCode EPSComputeVectors_QZ(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_DAVIDSON    *data = (EPS_DAVIDSON*)eps->data;
  dvdDashboard    *d = &data->ddb;
  PetscScalar     *pX, *auxS;
  PetscInt        size_auxS;
  
  PetscFunctionBegin;

  /* Finish cS and cT */
  ierr = VecsMultIb(d->cS, 0, d->ldcS, d->nconv, d->nconv, d->auxS, d->V[0]);
  CHKERRQ(ierr);
  if (d->cT) {
    ierr = VecsMultIb(d->cT, 0, d->ldcT, d->nconv, d->nconv, d->auxS, d->V[0]);
    CHKERRQ(ierr);
  }

  /* Compute the eigenvectors associated to (cS, cT) */
  ierr = PetscMalloc(sizeof(PetscScalar)*d->nconv*d->nconv, &pX); CHKERRQ(ierr);
  size_auxS = 11*d->nconv + 4*d->nconv*d->nconv; 
  ierr = PetscMalloc(sizeof(PetscScalar)*size_auxS, &auxS); CHKERRQ(ierr);
  ierr = dvd_compute_eigenvectors(d->nconv, d->cS, d->ldcS, d->cT, d->ldcT,
                                  pX, d->nconv, PETSC_NULL, 0, auxS,
                                  size_auxS, PETSC_FALSE); CHKERRQ(ierr);

  /* pX[i] <- pX[i] / ||pX[i]|| */
  ierr = SlepcDenseNorm(pX, d->nconv, d->nconv, d->nconv, d->ceigi);
  CHKERRQ(ierr);

  /* V <- cX * pX */ 
  ierr = SlepcUpdateVectorsZ(eps->V, 0.0, 1.0, d->cX, d->size_cX, pX,
                             d->nconv, d->nconv, d->nconv); CHKERRQ(ierr);

  ierr = PetscFree(pX); CHKERRQ(ierr);
  ierr = PetscFree(auxS); CHKERRQ(ierr);

  eps->evecsavailable = PETSC_TRUE;

  PetscFunctionReturn(0);
}
EXTERN_C_END
