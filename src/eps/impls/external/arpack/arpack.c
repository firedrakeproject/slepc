/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to the ARPACK package
*/

#include <slepc/private/epsimpl.h>
#include "arpack.h"

PetscErrorCode EPSSetUp_ARPACK(EPS eps)
{
  PetscInt       ncv;
  EPS_ARPACK     *ar = (EPS_ARPACK*)eps->data;

  PetscFunctionBegin;
  EPSCheckDefinite(eps);
  if (eps->ncv!=PETSC_DEFAULT) {
    PetscCheck(eps->ncv>=eps->nev+2,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The value of ncv must be at least nev+2");
  } else eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),eps->n); /* set default value of ncv */
  if (eps->mpd!=PETSC_DEFAULT) CHKERRQ(PetscInfo(eps,"Warning: parameter mpd ignored\n"));
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = PetscMax(300,(PetscInt)(2*eps->n/eps->ncv));
  if (!eps->which) CHKERRQ(EPSSetWhichEigenpairs_Default(eps));
  PetscCheck(eps->which!=EPS_ALL,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues");
  PetscCheck(eps->which!=EPS_WHICH_USER,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support user-defined ordering of eigenvalues");
  EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_CONVERGENCE | EPS_FEATURE_STOPPING | EPS_FEATURE_TWOSIDED);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION);

  ncv = eps->ncv;
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscFree(ar->rwork));
  CHKERRQ(PetscMalloc1(ncv,&ar->rwork));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,ncv*sizeof(PetscReal)));
  ar->lworkl = 3*ncv*ncv+5*ncv;
  CHKERRQ(PetscFree(ar->workev));
  CHKERRQ(PetscMalloc1(3*ncv,&ar->workev));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,3*ncv*sizeof(PetscScalar)));
#else
  if (eps->ishermitian) {
    ar->lworkl = ncv*(ncv+8);
  } else {
    ar->lworkl = 3*ncv*ncv+6*ncv;
    CHKERRQ(PetscFree(ar->workev));
    CHKERRQ(PetscMalloc1(3*ncv,&ar->workev));
    CHKERRQ(PetscLogObjectMemory((PetscObject)eps,3*ncv*sizeof(PetscScalar)));
  }
#endif
  CHKERRQ(PetscFree(ar->workl));
  CHKERRQ(PetscMalloc1(ar->lworkl,&ar->workl));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,ar->lworkl*sizeof(PetscScalar)));
  CHKERRQ(PetscFree(ar->select));
  CHKERRQ(PetscMalloc1(ncv,&ar->select));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,ncv*sizeof(PetscInt)));
  CHKERRQ(PetscFree(ar->workd));
  CHKERRQ(PetscMalloc1(3*eps->nloc,&ar->workd));
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,3*eps->nloc*sizeof(PetscScalar)));

  CHKERRQ(EPSAllocateSolution(eps,0));
  CHKERRQ(EPS_SetInnerProduct(eps));
  CHKERRQ(EPSSetWorkVecs(eps,2));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_ARPACK(EPS eps)
{
  EPS_ARPACK     *ar = (EPS_ARPACK*)eps->data;
  char           bmat[1],howmny[] = "A";
  const char     *which;
  PetscInt       n,iparam[11],ipntr[14],ido,info,nev,ncv,rvec;
#if !defined(PETSC_HAVE_MPIUNI) && !defined(PETSC_HAVE_MSMPI)
  MPI_Fint       fcomm;
#endif
  PetscScalar    sigmar,*pV,*resid;
  Vec            x,y,w = eps->work[0];
  Mat            A;
  PetscBool      isSinv,isShift;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    sigmai = 0.0;
#endif

  PetscFunctionBegin;
  nev = eps->nev;
  ncv = eps->ncv;
#if !defined(PETSC_HAVE_MPIUNI) && !defined(PETSC_HAVE_MSMPI)
  fcomm = MPI_Comm_c2f(PetscObjectComm((PetscObject)eps));
#endif
  n = eps->nloc;
  CHKERRQ(EPSGetStartVector(eps,0,NULL));
  CHKERRQ(BVSetActiveColumns(eps->V,0,0));  /* just for deflation space */
  CHKERRQ(BVCopyVec(eps->V,0,eps->work[1]));
  CHKERRQ(BVGetArray(eps->V,&pV));
  CHKERRQ(VecGetArray(eps->work[1],&resid));

  ido  = 0;            /* first call to reverse communication interface */
  info = 1;            /* indicates an initial vector is provided */
  iparam[0] = 1;       /* use exact shifts */
  iparam[2] = eps->max_it;  /* max Arnoldi iterations */
  iparam[3] = 1;       /* blocksize */
  iparam[4] = 0;       /* number of converged Ritz values */

  /*
     Computational modes ([]=not supported):
            symmetric    non-symmetric    complex
        1     1  'I'        1  'I'         1  'I'
        2     3  'I'        3  'I'         3  'I'
        3     2  'G'        2  'G'         2  'G'
        4     3  'G'        3  'G'         3  'G'
        5   [ 4  'G' ]    [ 3  'G' ]
        6   [ 5  'G' ]    [ 4  'G' ]
   */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&isSinv));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isShift));
  CHKERRQ(STGetShift(eps->st,&sigmar));
  CHKERRQ(STGetMatrix(eps->st,0,&A));
  CHKERRQ(MatCreateVecsEmpty(A,&x,&y));

  if (isSinv) {
    /* shift-and-invert mode */
    iparam[6] = 3;
    if (eps->ispositive) bmat[0] = 'G';
    else bmat[0] = 'I';
  } else if (isShift && eps->ispositive) {
    /* generalized shift mode with B positive definite */
    iparam[6] = 2;
    bmat[0] = 'G';
  } else {
    /* regular mode */
    PetscCheck(!eps->ishermitian || !eps->isgeneralized,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectral transformation not supported by ARPACK hermitian solver");
    iparam[6] = 1;
    bmat[0] = 'I';
  }

#if !defined(PETSC_USE_COMPLEX)
  if (eps->ishermitian) {
    switch (eps->which) {
      case EPS_TARGET_MAGNITUDE:
      case EPS_LARGEST_MAGNITUDE:  which = "LM"; break;
      case EPS_SMALLEST_MAGNITUDE: which = "SM"; break;
      case EPS_TARGET_REAL:
      case EPS_LARGEST_REAL:       which = "LA"; break;
      case EPS_SMALLEST_REAL:      which = "SA"; break;
      default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Wrong value of eps->which");
    }
  } else {
#endif
    switch (eps->which) {
      case EPS_TARGET_MAGNITUDE:
      case EPS_LARGEST_MAGNITUDE:  which = "LM"; break;
      case EPS_SMALLEST_MAGNITUDE: which = "SM"; break;
      case EPS_TARGET_REAL:
      case EPS_LARGEST_REAL:       which = "LR"; break;
      case EPS_SMALLEST_REAL:      which = "SR"; break;
      case EPS_TARGET_IMAGINARY:
      case EPS_LARGEST_IMAGINARY:  which = "LI"; break;
      case EPS_SMALLEST_IMAGINARY: which = "SI"; break;
      default: SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Wrong value of eps->which");
    }
#if !defined(PETSC_USE_COMPLEX)
  }
#endif

  do {

#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      PetscStackCall("ARPACKsaupd",ARPACKsaupd_(&fcomm,&ido,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,&info));
    } else {
      PetscStackCall("ARPACKnaupd",ARPACKnaupd_(&fcomm,&ido,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,&info));
    }
#else
    PetscStackCall("ARPACKnaupd",ARPACKnaupd_(&fcomm,&ido,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,ar->rwork,&info));
#endif

    if (ido == -1 || ido == 1 || ido == 2) {
      if (ido == 1 && iparam[6] == 3 && bmat[0] == 'G') {
        /* special case for shift-and-invert with B semi-positive definite*/
        CHKERRQ(VecPlaceArray(x,&ar->workd[ipntr[2]-1]));
      } else {
        CHKERRQ(VecPlaceArray(x,&ar->workd[ipntr[0]-1]));
      }
      CHKERRQ(VecPlaceArray(y,&ar->workd[ipntr[1]-1]));

      if (ido == -1) {
        /* Y = OP * X for for the initialization phase to
           force the starting vector into the range of OP */
        CHKERRQ(STApply(eps->st,x,y));
      } else if (ido == 2) {
        /* Y = B * X */
        CHKERRQ(BVApplyMatrix(eps->V,x,y));
      } else { /* ido == 1 */
        if (iparam[6] == 3 && bmat[0] == 'G') {
          /* Y = OP * X for shift-and-invert with B semi-positive definite */
          CHKERRQ(STMatSolve(eps->st,x,y));
        } else if (iparam[6] == 2) {
          /* X=A*X Y=B^-1*X for shift with B positive definite */
          CHKERRQ(MatMult(A,x,y));
          if (sigmar != 0.0) {
            CHKERRQ(BVApplyMatrix(eps->V,x,w));
            CHKERRQ(VecAXPY(y,sigmar,w));
          }
          CHKERRQ(VecCopy(y,x));
          CHKERRQ(STMatSolve(eps->st,x,y));
        } else {
          /* Y = OP * X */
          CHKERRQ(STApply(eps->st,x,y));
        }
        CHKERRQ(BVOrthogonalizeVec(eps->V,y,NULL,NULL,NULL));
      }

      CHKERRQ(VecResetArray(x));
      CHKERRQ(VecResetArray(y));
    } else PetscCheck(ido==99,PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Internal error in ARPACK reverse communication interface (ido=%" PetscInt_FMT ")",ido);

  } while (ido != 99);

  eps->nconv = iparam[4];
  eps->its = iparam[2];

  PetscCheck(info!=3,PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"No shift could be applied in xxAUPD.\nTry increasing the size of NCV relative to NEV");
  PetscCheck(info==0 || info==1,PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxAUPD (%" PetscInt_FMT ")",info);

  rvec = PETSC_TRUE;

  if (eps->nconv > 0) {
#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      PetscStackCall("ARPACKseupd",ARPACKseupd_(&fcomm,&rvec,howmny,ar->select,eps->eigr,pV,&n,&sigmar,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,&info));
    } else {
      PetscStackCall("ARPACKneupd",ARPACKneupd_(&fcomm,&rvec,howmny,ar->select,eps->eigr,eps->eigi,pV,&n,&sigmar,&sigmai,ar->workev,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,&info));
    }
#else
    PetscStackCall("ARPACKneupd",ARPACKneupd_(&fcomm,&rvec,howmny,ar->select,eps->eigr,pV,&n,&sigmar,ar->workev,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,ar->rwork,&info));
#endif
    PetscCheck(info==0,PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxEUPD (%" PetscInt_FMT ")",info);
  }

  CHKERRQ(BVRestoreArray(eps->V,&pV));
  CHKERRQ(VecRestoreArray(eps->work[1],&resid));
  if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSBackTransform_ARPACK(EPS eps)
{
  PetscBool      isSinv;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&isSinv));
  if (!isSinv) {
    CHKERRQ(EPSBackTransform_Default(eps));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_ARPACK(EPS eps)
{
  EPS_ARPACK     *ar = (EPS_ARPACK*)eps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(ar->workev));
  CHKERRQ(PetscFree(ar->workl));
  CHKERRQ(PetscFree(ar->select));
  CHKERRQ(PetscFree(ar->workd));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscFree(ar->rwork));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_ARPACK(EPS eps)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(eps->data));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_ARPACK(EPS eps)
{
  EPS_ARPACK     *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(eps,&ctx));
  eps->data = (void*)ctx;

  eps->ops->solve          = EPSSolve_ARPACK;
  eps->ops->setup          = EPSSetUp_ARPACK;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->destroy        = EPSDestroy_ARPACK;
  eps->ops->reset          = EPSReset_ARPACK;
  eps->ops->backtransform  = EPSBackTransform_ARPACK;
  PetscFunctionReturn(0);
}
