/*
   Skeleton of Davidson solver. Actual solvers are GD and JD.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "davidson.h"

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@Article{slepc-davidson,\n"
  "   author = \"E. Romero and J. E. Roman\",\n"
  "   title = \"A parallel implementation of {Davidson} methods for large-scale eigenvalue problems in {SLEPc}\",\n"
  "   journal = \"{ACM} Trans. Math. Software\",\n"
  "   volume = \"40\",\n"
  "   number = \"2\",\n"
  "   pages = \"13:1--13:29\",\n"
  "   year = \"2014,\"\n"
  "   doi = \"http://dx.doi.org/10.1145/2543696\"\n"
  "}\n";

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_XD"
PetscErrorCode EPSSetUp_XD(EPS eps)
{
  PetscErrorCode ierr;
  EPS_DAVIDSON   *data = (EPS_DAVIDSON*)eps->data;
  dvdDashboard   *dvd = &data->ddb;
  dvdBlackboard  b;
  PetscInt       min_size_V,plusk,bs,initv,i,cX_in_proj,cX_in_impr,nmat;
  Mat            A,B;
  KSP            ksp;
  PetscBool      t,ipB,ispositive,dynamic;
  HarmType_t     harm;
  InitType_t     init;
  PetscReal      fix;
  PetscScalar    target;

  PetscFunctionBegin;
  /* Setup EPS options and get the problem specification */
  ierr = EPSXDGetBlockSize_XD(eps,&bs);CHKERRQ(ierr);
  if (bs <= 0) bs = 1;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The value of ncv must be at least nev");
  } else if (eps->mpd) eps->ncv = eps->mpd + eps->nev + bs;
  else if (eps->nev<500) eps->ncv = PetscMin(eps->n-bs,PetscMax(2*eps->nev,eps->nev+15))+bs;
  else eps->ncv = PetscMin(eps->n-bs,eps->nev+500)+bs;
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (eps->mpd > eps->ncv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The mpd has to be less or equal than ncv");
  if (eps->mpd < 2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The mpd has to be greater than 2");
  if (!eps->max_it) eps->max_it = PetscMax(100*eps->ncv,2*eps->n);
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY)) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Wrong value of eps->which");
  if (!(eps->nev + bs <= eps->ncv)) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The ncv has to be greater than nev plus blocksize");
  if (eps->trueres) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"-eps_true_residual is temporally disable in this solver.");

  ierr = EPSXDGetRestart_XD(eps,&min_size_V,&plusk);CHKERRQ(ierr);
  if (!min_size_V) min_size_V = PetscMin(PetscMax(bs,5),eps->mpd/2);
  if (!(min_size_V+bs <= eps->mpd)) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The value of minv must be less than mpd minus blocksize");
  ierr = EPSXDGetInitialSize_XD(eps,&initv);CHKERRQ(ierr);
  if (eps->mpd < initv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The initv has to be less or equal than mpd");

  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_FALSE);CHKERRQ(ierr);

  /* Change the default sigma to inf if necessary */
  if (eps->which == EPS_LARGEST_MAGNITUDE || eps->which == EPS_LARGEST_REAL || eps->which == EPS_LARGEST_IMAGINARY) {
    ierr = STSetDefaultShift(eps->st,PETSC_MAX_REAL);CHKERRQ(ierr);
  }

  /* Davidson solvers only support STPRECOND */
  ierr = STSetUp(eps->st);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STPRECOND,&t);CHKERRQ(ierr);
  if (!t) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"%s only works with precond spectral transformation",((PetscObject)eps)->type_name);

  /* Setup problem specification in dvd */
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  ierr = EPSReset_XD(eps);CHKERRQ(ierr);
  ierr = PetscMemzero(dvd,sizeof(dvdDashboard));CHKERRQ(ierr);
  dvd->A = A; dvd->B = eps->isgeneralized? B: NULL;
  ispositive = eps->ispositive;
  dvd->sA = DVD_MAT_IMPLICIT | (eps->ishermitian? DVD_MAT_HERMITIAN: 0) | ((ispositive && !eps->isgeneralized) ? DVD_MAT_POS_DEF: 0);
  /* Asume -eps_hermitian means hermitian-definite in generalized problems */
  if (!ispositive && !eps->isgeneralized && eps->ishermitian) ispositive = PETSC_TRUE;
  if (!eps->isgeneralized) dvd->sB = DVD_MAT_IMPLICIT | DVD_MAT_HERMITIAN | DVD_MAT_IDENTITY | DVD_MAT_UNITARY | DVD_MAT_POS_DEF;
  else dvd->sB = DVD_MAT_IMPLICIT | (eps->ishermitian? DVD_MAT_HERMITIAN: 0) | (ispositive? DVD_MAT_POS_DEF: 0);
  ipB = (dvd->B && data->ipB && DVD_IS(dvd->sB,DVD_MAT_HERMITIAN))?PETSC_TRUE:PETSC_FALSE;
  if (data->ipB && !ipB) data->ipB = PETSC_FALSE;
  dvd->correctXnorm = ipB;
  dvd->sEP = ((!eps->isgeneralized || (eps->isgeneralized && ipB))? DVD_EP_STD: 0) | (ispositive? DVD_EP_HERMITIAN: 0) | ((eps->problem_type == EPS_GHIEP && ipB) ? DVD_EP_INDEFINITE : 0);
  dvd->nev        = eps->nev;
  dvd->which      = eps->which;
  dvd->withTarget = PETSC_TRUE;
  switch (eps->which) {
    case EPS_TARGET_MAGNITUDE:
    case EPS_TARGET_IMAGINARY:
      dvd->target[0] = target = eps->target;
      dvd->target[1] = 1.0;
      break;
    case EPS_TARGET_REAL:
      dvd->target[0] = PetscRealPart(target = eps->target);
      dvd->target[1] = 1.0;
      break;
    case EPS_LARGEST_REAL:
    case EPS_LARGEST_MAGNITUDE:
    case EPS_LARGEST_IMAGINARY: /* TODO: think about this case */
      dvd->target[0] = 1.0;
      dvd->target[1] = target = 0.0;
      break;
    case EPS_SMALLEST_MAGNITUDE:
    case EPS_SMALLEST_REAL:
    case EPS_SMALLEST_IMAGINARY: /* TODO: think about this case */
      dvd->target[0] = target = 0.0;
      dvd->target[1] = 1.0;
      break;
    case EPS_WHICH_USER:
      ierr = STGetShift(eps->st,&target);CHKERRQ(ierr);
      dvd->target[0] = target;
      dvd->target[1] = 1.0;
      break;
    case EPS_ALL:
      SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported option: which == EPS_ALL");
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported value of option 'which'");
  }
  dvd->tol = (eps->tol==PETSC_DEFAULT)? SLEPC_DEFAULT_TOL: eps->tol;
  dvd->eps = eps;

  /* Setup the extraction technique */
  if (!eps->extraction) {
    if (ipB || ispositive) eps->extraction = EPS_RITZ;
    else {
      switch (eps->which) {
        case EPS_TARGET_REAL:
        case EPS_TARGET_MAGNITUDE:
        case EPS_TARGET_IMAGINARY:
        case EPS_SMALLEST_MAGNITUDE:
        case EPS_SMALLEST_REAL:
        case EPS_SMALLEST_IMAGINARY:
          eps->extraction = EPS_HARMONIC;
          break;
        case EPS_LARGEST_REAL:
        case EPS_LARGEST_MAGNITUDE:
        case EPS_LARGEST_IMAGINARY:
          eps->extraction = EPS_HARMONIC_LARGEST;
          break;
        default:
          eps->extraction = EPS_RITZ;
      }
    }
  }
  switch (eps->extraction) {
    case EPS_RITZ:              harm = DVD_HARM_NONE; break;
    case EPS_HARMONIC:          harm = DVD_HARM_RR; break;
    case EPS_HARMONIC_RELATIVE: harm = DVD_HARM_RRR; break;
    case EPS_HARMONIC_RIGHT:    harm = DVD_HARM_REIGS; break;
    case EPS_HARMONIC_LARGEST:  harm = DVD_HARM_LEIGS; break;
    default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
  }

  /* Setup the type of starting subspace */
  ierr = EPSXDGetKrylovStart_XD(eps,&t);CHKERRQ(ierr);
  init = (!t)? DVD_INITV_CLASSIC : DVD_INITV_KRYLOV;

  /* Setup the presence of converged vectors in the projected problem and the projector */
  ierr = EPSXDGetWindowSizes_XD(eps,&cX_in_impr,&cX_in_proj);CHKERRQ(ierr);
  if (cX_in_impr>0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The option pwindow is temporally disable in this solver.");
  if (cX_in_proj>0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"The option qwindow is temporally disable in this solver.");
  if (min_size_V <= cX_in_proj) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"minv has to be greater than qwindow");
  if (bs > 1 && cX_in_impr > 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported option: pwindow > 0 and bs > 1");

  /* Get the fix parameter */
  ierr = EPSXDGetFix_XD(eps,&fix);CHKERRQ(ierr);

  /* Get whether the stopping criterion is used */
  ierr = EPSJDGetConstCorrectionTol_JD(eps,&dynamic);CHKERRQ(ierr);

  /* Preconfigure dvd */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  ierr = dvd_schm_basic_preconf(dvd,&b,eps->mpd,min_size_V,bs,initv,PetscAbs(eps->nini),plusk,harm,ksp,init,eps->trackall,data->ipB,cX_in_proj,cX_in_impr,data->doubleexp);CHKERRQ(ierr);

  /* Allocate memory */
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);

  /* Setup orthogonalization */
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  if (!(ipB && dvd->B)) {
    ierr = BVSetMatrix(eps->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
  }

  for (i=0;i<eps->ncv;i++) eps->perm[i] = i;

  /* Configure dvd for a basic GD */
  ierr = dvd_schm_basic_conf(dvd,&b,eps->mpd,min_size_V,bs,initv,PetscAbs(eps->nini),plusk,harm,dvd->withTarget,target,ksp,fix,init,eps->trackall,data->ipB,cX_in_proj,cX_in_impr,dynamic,data->doubleexp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_XD"
PetscErrorCode EPSSolve_XD(EPS eps)
{
  EPS_DAVIDSON   *data = (EPS_DAVIDSON*)eps->data;
  dvdDashboard   *d = &data->ddb;
  PetscInt       l,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  /* Call the starting routines */
  ierr = EPSDavidsonFLCall(d->startList,d);CHKERRQ(ierr);

  while (eps->reason == EPS_CONVERGED_ITERATING) {

    /* Initialize V, if it is needed */
    ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
    if (l == k) { ierr = d->initV(d);CHKERRQ(ierr); }

    /* Find the best approximated eigenpairs in V, X */
    ierr = d->calcPairs(d);CHKERRQ(ierr);

    /* Test for convergence */
    ierr = (*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx);CHKERRQ(ierr);
    if (eps->reason != EPS_CONVERGED_ITERATING) break;

    /* Expand the subspace */
    ierr = d->updateV(d);CHKERRQ(ierr);

    /* Monitor progress */
    eps->nconv = d->nconv;
    eps->its++;
    ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,k);CHKERRQ(ierr);
  }

  /* Call the ending routines */
  ierr = EPSDavidsonFLCall(d->endList,d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_XD"
PetscErrorCode EPSReset_XD(EPS eps)
{
  EPS_DAVIDSON   *data = (EPS_DAVIDSON*)eps->data;
  dvdDashboard   *dvd = &data->ddb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Call step destructors and destroys the list */
  ierr = EPSDavidsonFLCall(dvd->destroyList,dvd);CHKERRQ(ierr);
  ierr = EPSDavidsonFLDestroy(&dvd->destroyList);CHKERRQ(ierr);
  ierr = EPSDavidsonFLDestroy(&dvd->startList);CHKERRQ(ierr);
  ierr = EPSDavidsonFLDestroy(&dvd->endList);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDSetKrylovStart_XD"
PetscErrorCode EPSXDSetKrylovStart_XD(EPS eps,PetscBool krylovstart)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  data->krylovstart = krylovstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetKrylovStart_XD"
PetscErrorCode EPSXDGetKrylovStart_XD(EPS eps,PetscBool *krylovstart)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *krylovstart = data->krylovstart;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDSetBlockSize_XD"
PetscErrorCode EPSXDSetBlockSize_XD(EPS eps,PetscInt blocksize)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (blocksize == PETSC_DEFAULT || blocksize == PETSC_DECIDE) blocksize = 1;
  if (blocksize <= 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid blocksize value");
  data->blocksize = blocksize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetBlockSize_XD"
PetscErrorCode EPSXDGetBlockSize_XD(EPS eps,PetscInt *blocksize)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *blocksize = data->blocksize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDSetRestart_XD"
PetscErrorCode EPSXDSetRestart_XD(EPS eps,PetscInt minv,PetscInt plusk)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (minv == PETSC_DEFAULT || minv == PETSC_DECIDE) minv = 5;
  if (minv <= 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid minv value");
  if (plusk == PETSC_DEFAULT || plusk == PETSC_DECIDE) plusk = 5;
  if (plusk < 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid plusk value");
  data->minv = minv;
  data->plusk = plusk;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetRestart_XD"
PetscErrorCode EPSXDGetRestart_XD(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (minv) *minv = data->minv;
  if (plusk) *plusk = data->plusk;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetInitialSize_XD"
PetscErrorCode EPSXDGetInitialSize_XD(EPS eps,PetscInt *initialsize)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *initialsize = data->initialsize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDSetInitialSize_XD"
PetscErrorCode EPSXDSetInitialSize_XD(EPS eps,PetscInt initialsize)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (initialsize == PETSC_DEFAULT || initialsize == PETSC_DECIDE) initialsize = 5;
  if (initialsize <= 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid initial size value");
  data->initialsize = initialsize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetFix_XD"
PetscErrorCode EPSXDGetFix_XD(EPS eps,PetscReal *fix)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *fix = data->fix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetFix_JD"
PetscErrorCode EPSJDSetFix_JD(EPS eps,PetscReal fix)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (fix == PETSC_DEFAULT || fix == PETSC_DECIDE) fix = 0.01;
  if (fix < 0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid fix value");
  data->fix = fix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDSetBOrth_XD"
PetscErrorCode EPSXDSetBOrth_XD(EPS eps,PetscBool borth)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  data->ipB = borth;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetBOrth_XD"
PetscErrorCode EPSXDGetBOrth_XD(EPS eps,PetscBool *borth)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *borth = data->ipB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetConstCorrectionTol_JD"
PetscErrorCode EPSJDSetConstCorrectionTol_JD(EPS eps,PetscBool constant)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  data->dynamic = PetscNot(constant);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetConstCorrectionTol_JD"
PetscErrorCode EPSJDGetConstCorrectionTol_JD(EPS eps,PetscBool *constant)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *constant = PetscNot(data->dynamic);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDSetWindowSizes_XD"
PetscErrorCode EPSXDSetWindowSizes_XD(EPS eps,PetscInt pwindow,PetscInt qwindow)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (pwindow == PETSC_DEFAULT || pwindow == PETSC_DECIDE) pwindow = 0;
  if (pwindow < 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid pwindow value");
  if (qwindow == PETSC_DEFAULT || qwindow == PETSC_DECIDE) qwindow = 0;
  if (qwindow < 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid qwindow value");
  data->cX_in_proj = qwindow;
  data->cX_in_impr = pwindow;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSXDGetWindowSizes_XD"
PetscErrorCode EPSXDGetWindowSizes_XD(EPS eps,PetscInt *pwindow,PetscInt *qwindow)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  if (pwindow) *pwindow = data->cX_in_impr;
  if (qwindow) *qwindow = data->cX_in_proj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSComputeVectors_XD"
/*
  EPSComputeVectors_XD - Compute eigenvectors from the vectors
  provided by the eigensolver. This version is intended for solvers
  that provide Schur vectors from the QZ decomposition. Given the partial
  Schur decomposition OP*V=V*T, the following steps are performed:
      1) compute eigenvectors of (S,T): S*Z=T*Z*D
      2) compute eigenvectors of OP: X=V*Z
 */
PetscErrorCode EPSComputeVectors_XD(EPS eps)
{
  PetscErrorCode ierr;
  Mat            X;
  PetscBool      symm;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)eps->ds,DSHEP,&symm);CHKERRQ(ierr);
  if (symm) PetscFunctionReturn(0);
  ierr = DSVectors(eps->ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);

  /* V <- V * X */
  ierr = DSGetMat(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(eps->V,0,eps->nconv);CHKERRQ(ierr);
  ierr = BVMultInPlace(eps->V,X,0,eps->nconv);CHKERRQ(ierr);
  ierr = DSRestoreMat(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
