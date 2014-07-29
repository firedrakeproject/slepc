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
#include <slepc-private/mfnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define mfndestroy_                 MFNDESTROY
#define mfnview_                    MFNVIEW
#define mfnsetoptionsprefix_        MFNSETOPTIONSPREFIX
#define mfnappendoptionsprefix_     MFNAPPENDOPTIONSPREFIX
#define mfngetoptionsprefix_        MFNGETOPTIONSPREFIX
#define mfncreate_                  MFNCREATE
#define mfnsettype_                 MFNSETTYPE
#define mfngettype_                 MFNGETTYPE
#define mfnmonitordefault_          MFNMONITORDEFAULT
#define mfnmonitorlg_               MFNMONITORLG
#define mfnmonitorset_              MFNMONITORSET
#define mfngetbv_                   MFNGETBV
#define mfngetds_                   MFNGETDS
#define mfngetconvergedreason_      MFNGETCONVERGEDREASON
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define mfndestroy_                 mfndestroy
#define mfnview_                    mfnview
#define mfnsetoptionsprefix_        mfnsetoptionsprefix
#define mfnappendoptionsprefix_     mfnappendoptionsprefix
#define mfngetoptionsprefix_        mfngetoptionsprefix
#define mfncreate_                  mfncreate
#define mfnsettype_                 mfnsettype
#define mfngettype_                 mfngettype
#define mfnmonitordefault_          mfnmonitordefault
#define mfnmonitorlg_               mfnmonitorlg
#define mfnmonitorset_              mfnmonitorset
#define mfngetbv_                   mfngetbv
#define mfngetds_                   mfngetds
#define mfngetconvergedreason_      mfngetconvergedreason
#endif

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code, hence no STDCALL
*/
PETSC_EXTERN void mfnmonitordefault_(MFN *mfn,PetscInt *it,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = MFNMonitorDefault(*mfn,*it,*errest,ctx);
}

PETSC_EXTERN void mfnmonitorlg_(MFN *mfn,PetscInt *it,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = MFNMonitorLG(*mfn,*it,*errest,ctx);
}

static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
#undef __FUNCT__
#define __FUNCT__ "ourmonitor"
static PetscErrorCode ourmonitor(MFN mfn,PetscInt i,PetscReal d,void* ctx)
{
  PetscObjectUseFortranCallback(mfn,_cb.monitor,(MFN*,PetscInt*,PetscReal*,void*,PetscErrorCode*),(&mfn,&i,&d,_ctx,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourdestroy"
static PetscErrorCode ourdestroy(void** ctx)
{
  MFN mfn = (MFN)*ctx;
  PetscObjectUseFortranCallback(mfn,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL mfndestroy_(MFN *mfn,PetscErrorCode *ierr)
{
  *ierr = MFNDestroy(mfn);
}

PETSC_EXTERN void PETSC_STDCALL mfnview_(MFN *mfn,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MFNView(*mfn,v);
}

PETSC_EXTERN void PETSC_STDCALL mfnsettype_(MFN *mfn,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MFNSetType(*mfn,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL mfngettype_(MFN *mfn,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  MFNType tname;

  *ierr = MFNGetType(*mfn,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL mfnsetoptionsprefix_(MFN *mfn,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = MFNSetOptionsPrefix(*mfn,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL mfnappendoptionsprefix_(MFN *mfn,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = MFNAppendOptionsPrefix(*mfn,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL mfngetoptionsprefix_(MFN *mfn,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = MFNGetOptionsPrefix(*mfn,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

PETSC_EXTERN void PETSC_STDCALL mfncreate_(MPI_Fint *comm,MFN *mfn,PetscErrorCode *ierr)
{
  *ierr = MFNCreate(MPI_Comm_f2c(*(comm)),mfn);
}

PETSC_EXTERN void PETSC_STDCALL mfnmonitorset_(MFN *mfn,void (PETSC_STDCALL *monitor)(MFN*,PetscInt*,PetscReal*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)mfnmonitordefault_) {
    *ierr = MFNMonitorSet(*mfn,MFNMonitorDefault,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)mfnmonitorlg_) {
    *ierr = MFNMonitorSet(*mfn,MFNMonitorLG,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*mfn,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    if (!monitordestroy) {
      *ierr = MFNMonitorSet(*mfn,ourmonitor,*mfn,0);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*mfn,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
      *ierr = MFNMonitorSet(*mfn,ourmonitor,*mfn,ourdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL mfngetbv_(MFN *mfn,BV *bv,PetscErrorCode *ierr)
{
  *ierr = MFNGetBV(*mfn,bv);
}

PETSC_EXTERN void PETSC_STDCALL mfngetds_(MFN *mfn,DS *ds,PetscErrorCode *ierr)
{
  *ierr = MFNGetDS(*mfn,ds);
}

PETSC_EXTERN void PETSC_STDCALL mfngetconvergedreason_(MFN *mfn,MFNConvergedReason *reason,PetscErrorCode *ierr)
{
  *ierr = MFNGetConvergedReason(*mfn,reason);
}

