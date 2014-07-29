/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <petsc-private/fortranimpl.h>
#include <slepc-private/slepcimpl.h>
#include <slepc-private/nepimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define nepdestroy_                 NEPDESTROY
#define nepview_                    NEPVIEW
#define nepsetoptionsprefix_        NEPSETOPTIONSPREFIX
#define nepappendoptionsprefix_     NEPAPPENDOPTIONSPREFIX
#define nepgetoptionsprefix_        NEPGETOPTIONSPREFIX
#define nepcreate_                  NEPCREATE
#define nepsettype_                 NEPSETTYPE
#define nepgettype_                 NEPGETTYPE
#define nepmonitorall_              NEPMONITORALL
#define nepmonitorlg_               NEPMONITORLG
#define nepmonitorlgall_            NEPMONITORLGALL
#define nepmonitorset_              NEPMONITORSET
#define nepmonitorconverged_        NEPMONITORCONVERGED
#define nepmonitorfirst_            NEPMONITORFIRST
#define nepgetbv_                   NEPGETBV
#define nepgetds_                   NEPGETDS
#define nepgetrg_                   NEPGETRG
#define nepgetksp                   NEPGETKSP
#define nepgetwhicheigenpairs_      NEPGETWHICHEIGENPAIRS
#define nepgetconvergedreason_      NEPGETCONVERGEDREASON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define nepdestroy_                 nepdestroy
#define nepview_                    nepview
#define nepsetoptionsprefix_        nepsetoptionsprefix
#define nepappendoptionsprefix_     nepappendoptionsprefix
#define nepgetoptionsprefix_        nepgetoptionsprefix
#define nepcreate_                  nepcreate
#define nepsettype_                 nepsettype
#define nepgettype_                 nepgettype
#define nepmonitorall_              nepmonitorall
#define nepmonitorlg_               nepmonitorlg
#define nepmonitorlgall_            nepmonitorlgall
#define nepmonitorset_              nepmonitorset
#define nepmonitorconverged_        nepmonitorconverged
#define nepmonitorfirst_            nepmonitorfirst
#define nepgetbv_                   nepgetbv
#define nepgetds_                   nepgetds
#define nepgetrg_                   nepgetrg
#define nepgetksp_                  nepgetksp
#define nepgetwhicheigenpairs_      nepgetwhicheigenpairs
#define nepgetconvergedreason_      nepgetconvergedreason
#endif

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code, hence no STDCALL
*/
PETSC_EXTERN void nepmonitorall_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eig,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorAll(*nep,*it,*nconv,eig,errest,*nest,ctx);
}

PETSC_EXTERN void nepmonitorlg_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eig,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorLG(*nep,*it,*nconv,eig,errest,*nest,ctx);
}

PETSC_EXTERN void nepmonitorlgall_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eig,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorLGAll(*nep,*it,*nconv,eig,errest,*nest,ctx);
}

PETSC_EXTERN void nepmonitorconverged_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eig,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorConverged(*nep,*it,*nconv,eig,errest,*nest,ctx);
}

PETSC_EXTERN void nepmonitorfirst_(NEP *nep,PetscInt *it,PetscInt *nconv,PetscScalar *eig,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = NEPMonitorFirst(*nep,*it,*nconv,eig,errest,*nest,ctx);
}

static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
#undef __FUNCT__
#define __FUNCT__ "ourmonitor"
static PetscErrorCode ourmonitor(NEP nep,PetscInt i,PetscInt nc,PetscScalar *eig,PetscReal *d,PetscInt l,void* ctx)
{
  PetscObjectUseFortranCallback(nep,_cb.monitor,(NEP*,PetscInt*,PetscInt*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),(&nep,&i,&nc,eig,d,&l,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourdestroy"
static PetscErrorCode ourdestroy(void** ctx)
{
  NEP nep = (NEP)*ctx;
  PetscObjectUseFortranCallback(nep,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL nepdestroy_(NEP *nep,PetscErrorCode *ierr)
{
  *ierr = NEPDestroy(nep);
}

PETSC_EXTERN void PETSC_STDCALL nepview_(NEP *nep,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = NEPView(*nep,v);
}

PETSC_EXTERN void PETSC_STDCALL nepsettype_(NEP *nep,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = NEPSetType(*nep,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL nepgettype_(NEP *nep,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  NEPType tname;

  *ierr = NEPGetType(*nep,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL nepsetoptionsprefix_(NEP *nep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = NEPSetOptionsPrefix(*nep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL nepappendoptionsprefix_(NEP *nep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = NEPAppendOptionsPrefix(*nep,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL nepcreate_(MPI_Fint *comm,NEP *nep,PetscErrorCode *ierr)
{
  *ierr = NEPCreate(MPI_Comm_f2c(*(comm)),nep);
}

PETSC_EXTERN void PETSC_STDCALL nepgetoptionsprefix_(NEP *nep,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = NEPGetOptionsPrefix(*nep,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

PETSC_EXTERN void PETSC_STDCALL nepmonitorset_(NEP *nep,void (PETSC_STDCALL *monitor)(NEP*,PetscInt*,PetscInt*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  SlepcConvMonitor ctx;

  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorall_) {
    *ierr = NEPMonitorSet(*nep,NEPMonitorAll,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorlg_) {
    *ierr = NEPMonitorSet(*nep,NEPMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorlgall_) {
    *ierr = NEPMonitorSet(*nep,NEPMonitorLGAll,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorconverged_) {
    if (mctx) {
      PetscError(PetscObjectComm((PetscObject)*nep),__LINE__,"nepmonitorset_",__FILE__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL,"Must provide PETSC_NULL_OBJECT as a context in the Fortran interface to NEPMonitorSet");
      *ierr = 1;
      return;
    }
    *ierr = PetscNew(&ctx);
    if (*ierr) return;
    ctx->viewer = NULL;
    *ierr = NEPMonitorSet(*nep,NEPMonitorConverged,ctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)nepmonitorfirst_) {
    *ierr = NEPMonitorSet(*nep,NEPMonitorFirst,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    if (!monitordestroy) {
      *ierr = NEPMonitorSet(*nep,ourmonitor,*nep,0);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
      *ierr = NEPMonitorSet(*nep,ourmonitor,*nep,ourdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL nepgetbv_(NEP *nep,BV *bv,PetscErrorCode *ierr)
{
  *ierr = NEPGetBV(*nep,bv);
}

PETSC_EXTERN void PETSC_STDCALL nepgetds_(NEP *nep,DS *ds,PetscErrorCode *ierr)
{
  *ierr = NEPGetDS(*nep,ds);
}

PETSC_EXTERN void PETSC_STDCALL nepgetrg_(NEP *nep,RG *rg,PetscErrorCode *ierr)
{
  *ierr = NEPGetRG(*nep,rg);
}

PETSC_EXTERN void PETSC_STDCALL nepgetksp_(NEP *nep,KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = NEPGetKSP(*nep,ksp);
}

PETSC_EXTERN void PETSC_STDCALL nepgetwhicheigenpairs_(NEP *nep,NEPWhich *which,PetscErrorCode *ierr)
{
  *ierr = NEPGetWhichEigenpairs(*nep,which);
}

PETSC_EXTERN void PETSC_STDCALL nepgetconvergedreason_(NEP *nep,NEPConvergedReason *reason,PetscErrorCode *ierr)
{
  *ierr = NEPGetConvergedReason(*nep,reason);
}

