/*
   This file implements a wrapper to the ARPACK package

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

#include <slepc/private/epsimpl.h>
#include <../src/eps/impls/external/arpack/arpackp.h>

PetscErrorCode EPSSolve_ARPACK(EPS);

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_ARPACK"
PetscErrorCode EPSSetUp_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       ncv;
  PetscBool      flg,istrivial;
  EPS_ARPACK     *ar = (EPS_ARPACK*)eps->data;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev+2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The value of ncv must be at least nev+2");
  } else eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),eps->n); /* set default value of ncv */
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(300,(PetscInt)(2*eps->n/eps->ncv));
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;

  ncv = eps->ncv;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(ar->rwork);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncv,&ar->rwork);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,ncv*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscBLASIntCast(3*ncv*ncv+5*ncv,&ar->lworkl);CHKERRQ(ierr);
  ierr = PetscFree(ar->workev);CHKERRQ(ierr);
  ierr = PetscMalloc1(3*ncv,&ar->workev);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,3*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
#else
  if (eps->ishermitian) {
    ierr = PetscBLASIntCast(ncv*(ncv+8),&ar->lworkl);CHKERRQ(ierr);
  } else {
    ierr = PetscBLASIntCast(3*ncv*ncv+6*ncv,&ar->lworkl);CHKERRQ(ierr);
    ierr = PetscFree(ar->workev);CHKERRQ(ierr);
    ierr = PetscMalloc1(3*ncv,&ar->workev);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)eps,3*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  }
#endif
  ierr = PetscFree(ar->workl);CHKERRQ(ierr);
  ierr = PetscMalloc1(ar->lworkl,&ar->workl);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,ar->lworkl*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscFree(ar->select);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncv,&ar->select);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,ncv*sizeof(PetscBool));CHKERRQ(ierr);
  ierr = PetscFree(ar->workd);CHKERRQ(ierr);
  ierr = PetscMalloc1(3*eps->nloc,&ar->workd);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,3*eps->nloc*sizeof(PetscScalar));CHKERRQ(ierr);

  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }

  if (eps->balance!=EPS_BALANCE_NONE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Balancing not supported in the Arpack interface");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  if (eps->stopping!=EPSStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"External packages do not support user-defined stopping test");

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,2);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)eps->V,BVVECS,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver requires a BV with contiguous storage");
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");

  /* dispatch solve method */
  eps->ops->solve = EPSSolve_ARPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_ARPACK"
PetscErrorCode EPSSolve_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARPACK     *ar = (EPS_ARPACK*)eps->data;
  char           bmat[1],howmny[] = "A";
  const char     *which;
  PetscBLASInt   n,iparam[11],ipntr[14],ido,info,nev,ncv;
#if !defined(PETSC_HAVE_MPIUNI)
  PetscBLASInt   fcomm;
#endif
  PetscScalar    sigmar,*pV,*resid;
  Vec            v0,x,y,w = eps->work[0];
  Mat            A;
  PetscBool      isSinv,isShift,rvec;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    sigmai = 0.0;
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(eps->nev,&nev);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(eps->ncv,&ncv);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_MPIUNI)
  ierr = PetscBLASIntCast(MPI_Comm_c2f(PetscObjectComm((PetscObject)eps)),&fcomm);CHKERRQ(ierr);
#endif
  ierr = PetscBLASIntCast(eps->nloc,&n);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)eps),1,eps->nloc,PETSC_DECIDE,NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)eps),1,eps->nloc,PETSC_DECIDE,NULL,&y);CHKERRQ(ierr);
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(eps->V,0,0);CHKERRQ(ierr);  /* just for deflation space */
  ierr = BVGetColumn(eps->V,0,&v0);CHKERRQ(ierr);
  ierr = VecCopy(v0,eps->work[1]);CHKERRQ(ierr);
  ierr = VecGetArray(v0,&pV);CHKERRQ(ierr);
  ierr = VecGetArray(eps->work[1],&resid);CHKERRQ(ierr);

  ido  = 0;            /* first call to reverse communication interface */
  info = 1;            /* indicates a initial vector is provided */
  iparam[0] = 1;       /* use exact shifts */
  ierr = PetscBLASIntCast(eps->max_it,&iparam[2]);CHKERRQ(ierr);  /* max Arnoldi iterations */
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
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&isSinv);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isShift);CHKERRQ(ierr);
  ierr = STGetShift(eps->st,&sigmar);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);

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
    if (eps->ishermitian && eps->isgeneralized)
      SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectral transformation not supported by ARPACK hermitian solver");
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
        ierr = VecPlaceArray(x,&ar->workd[ipntr[2]-1]);CHKERRQ(ierr);
      } else {
        ierr = VecPlaceArray(x,&ar->workd[ipntr[0]-1]);CHKERRQ(ierr);
      }
      ierr = VecPlaceArray(y,&ar->workd[ipntr[1]-1]);CHKERRQ(ierr);

      if (ido == -1) {
        /* Y = OP * X for for the initialization phase to
           force the starting vector into the range of OP */
        ierr = STApply(eps->st,x,y);CHKERRQ(ierr);
      } else if (ido == 2) {
        /* Y = B * X */
        ierr = BVApplyMatrix(eps->V,x,y);CHKERRQ(ierr);
      } else { /* ido == 1 */
        if (iparam[6] == 3 && bmat[0] == 'G') {
          /* Y = OP * X for shift-and-invert with B semi-positive definite */
          ierr = STMatSolve(eps->st,x,y);CHKERRQ(ierr);
        } else if (iparam[6] == 2) {
          /* X=A*X Y=B^-1*X for shift with B positive definite */
          ierr = MatMult(A,x,y);CHKERRQ(ierr);
          if (sigmar != 0.0) {
            ierr = BVApplyMatrix(eps->V,x,w);CHKERRQ(ierr);
            ierr = VecAXPY(y,sigmar,w);CHKERRQ(ierr);
          }
          ierr = VecCopy(y,x);CHKERRQ(ierr);
          ierr = STMatSolve(eps->st,x,y);CHKERRQ(ierr);
        } else {
          /* Y = OP * X */
          ierr = STApply(eps->st,x,y);CHKERRQ(ierr);
        }
        ierr = BVOrthogonalizeVec(eps->V,y,NULL,NULL,NULL);CHKERRQ(ierr);
      }

      ierr = VecResetArray(x);CHKERRQ(ierr);
      ierr = VecResetArray(y);CHKERRQ(ierr);
    } else if (ido != 99) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Internal error in ARPACK reverse comunication interface (ido=%d)",ido);

  } while (ido != 99);

  eps->nconv = iparam[4];
  eps->its = iparam[2];

  if (info==3) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"No shift could be applied in xxAUPD.\nTry increasing the size of NCV relative to NEV");
  else if (info!=0 && info!=1) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxAUPD (%d)",info);

  rvec = PETSC_TRUE;

  if (eps->nconv > 0) {
#if !defined(PETSC_USE_COMPLEX)
    if (eps->ishermitian) {
      ierr = EPSMonitor(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],eps->eigi,&ar->workl[ipntr[6]-1],eps->ncv);CHKERRQ(ierr);
      PetscStackCall("ARPACKseupd",ARPACKseupd_(&fcomm,&rvec,howmny,ar->select,eps->eigr,pV,&n,&sigmar,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,&info));
    } else {
      ierr = EPSMonitor(eps,iparam[2],iparam[4],&ar->workl[ipntr[5]-1],&ar->workl[ipntr[6]-1],&ar->workl[ipntr[7]-1],eps->ncv);CHKERRQ(ierr);
      PetscStackCall("ARPACKneupd",ARPACKneupd_(&fcomm,&rvec,howmny,ar->select,eps->eigr,eps->eigi,pV,&n,&sigmar,&sigmai,ar->workev,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,&info));
    }
#else
    ierr = EPSMonitor(eps,eps->its,iparam[4],&ar->workl[ipntr[5]-1],eps->eigi,(PetscReal*)&ar->workl[ipntr[7]-1],eps->ncv);CHKERRQ(ierr);
    PetscStackCall("ARPACKneupd",ARPACKneupd_(&fcomm,&rvec,howmny,ar->select,eps->eigr,pV,&n,&sigmar,ar->workev,bmat,&n,which,&nev,&eps->tol,resid,&ncv,pV,&n,iparam,ipntr,ar->workd,ar->workl,&ar->lworkl,ar->rwork,&info));
#endif
    if (info!=0) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by ARPACK subroutine xxEUPD (%d)",info);
  }

  ierr = VecRestoreArray(v0,&pV);CHKERRQ(ierr);
  ierr = BVRestoreColumn(eps->V,0,&v0);CHKERRQ(ierr);
  ierr = VecRestoreArray(eps->work[1],&resid);CHKERRQ(ierr);
  if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  if (eps->ishermitian) {
    ierr = PetscMemcpy(eps->errest,&ar->workl[ipntr[8]-1],eps->nconv);CHKERRQ(ierr);
  } else {
    ierr = PetscMemcpy(eps->errest,&ar->workl[ipntr[10]-1],eps->nconv);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSBackTransform_ARPACK"
PetscErrorCode EPSBackTransform_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      isSinv;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&isSinv);CHKERRQ(ierr);
  if (!isSinv) {
    ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_ARPACK"
PetscErrorCode EPSReset_ARPACK(EPS eps)
{
  PetscErrorCode ierr;
  EPS_ARPACK     *ar = (EPS_ARPACK*)eps->data;

  PetscFunctionBegin;
  ierr = PetscFree(ar->workev);CHKERRQ(ierr);
  ierr = PetscFree(ar->workl);CHKERRQ(ierr);
  ierr = PetscFree(ar->select);CHKERRQ(ierr);
  ierr = PetscFree(ar->workd);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(ar->rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_ARPACK"
PetscErrorCode EPSDestroy_ARPACK(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_ARPACK"
PETSC_EXTERN PetscErrorCode EPSCreate_ARPACK(EPS eps)
{
  EPS_ARPACK     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->setup                = EPSSetUp_ARPACK;
  eps->ops->destroy              = EPSDestroy_ARPACK;
  eps->ops->reset                = EPSReset_ARPACK;
  eps->ops->backtransform        = EPSBackTransform_ARPACK;
  PetscFunctionReturn(0);
}

