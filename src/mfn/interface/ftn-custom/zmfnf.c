/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepc/private/mfnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define mfnview_                          MFNVIEW
#define mfnreasonview_                    MFNREASONVIEW
#define mfnsetoptionsprefix_              MFNSETOPTIONSPREFIX
#define mfnappendoptionsprefix_           MFNAPPENDOPTIONSPREFIX
#define mfngetoptionsprefix_              MFNGETOPTIONSPREFIX
#define mfnsettype_                       MFNSETTYPE
#define mfngettype_                       MFNGETTYPE
#define mfnmonitordefault_                MFNMONITORDEFAULT
#define mfnmonitorlg_                     MFNMONITORLG
#define mfnmonitorset_                    MFNMONITORSET
#define mfngettolerances00_               MFNGETTOLERANCES00
#define mfngettolerances10_               MFNGETTOLERANCES10
#define mfngettolerances01_               MFNGETTOLERANCES01
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define mfnview_                          mfnview
#define mfnreasonview_                    mfnreasonview
#define mfnsetoptionsprefix_              mfnsetoptionsprefix
#define mfnappendoptionsprefix_           mfnappendoptionsprefix
#define mfngetoptionsprefix_              mfngetoptionsprefix
#define mfnsettype_                       mfnsettype
#define mfngettype_                       mfngettype
#define mfnmonitordefault_                mfnmonitordefault
#define mfnmonitorlg_                     mfnmonitorlg
#define mfnmonitorset_                    mfnmonitorset
#define mfngettolerances00_               mfngettolerances00
#define mfngettolerances10_               mfngettolerances10
#define mfngettolerances01_               mfngettolerances01
#endif

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code, hence no STDCALL
*/
SLEPC_EXTERN void mfnmonitordefault_(MFN *mfn,PetscInt *it,PetscReal *errest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = MFNMonitorDefault(*mfn,*it,*errest,*ctx);
}

SLEPC_EXTERN void mfnmonitorlg_(MFN *mfn,PetscInt *it,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = MFNMonitorLG(*mfn,*it,*errest,ctx);
}

static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmonitor(MFN mfn,PetscInt i,PetscReal d,void* ctx)
{
  PetscObjectUseFortranCallback(mfn,_cb.monitor,(MFN*,PetscInt*,PetscReal*,void*,PetscErrorCode*),(&mfn,&i,&d,_ctx,&ierr));
}

static PetscErrorCode ourdestroy(void** ctx)
{
  MFN mfn = (MFN)*ctx;
  PetscObjectUseFortranCallback(mfn,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

SLEPC_EXTERN void PETSC_STDCALL mfnview_(MFN *mfn,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MFNView(*mfn,v);
}

SLEPC_EXTERN void PETSC_STDCALL mfnreasonview_(MFN *mfn,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MFNReasonView(*mfn,v);
}

SLEPC_EXTERN void PETSC_STDCALL mfnsettype_(MFN *mfn,char *type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = MFNSetType(*mfn,t);if (*ierr) return;
  FREECHAR(type,t);
}

SLEPC_EXTERN void PETSC_STDCALL mfngettype_(MFN *mfn,char *name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  MFNType tname;

  *ierr = MFNGetType(*mfn,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

SLEPC_EXTERN void PETSC_STDCALL mfnsetoptionsprefix_(MFN *mfn,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = MFNSetOptionsPrefix(*mfn,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

SLEPC_EXTERN void PETSC_STDCALL mfnappendoptionsprefix_(MFN *mfn,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = MFNAppendOptionsPrefix(*mfn,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

SLEPC_EXTERN void PETSC_STDCALL mfngetoptionsprefix_(MFN *mfn,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = MFNGetOptionsPrefix(*mfn,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

SLEPC_EXTERN void PETSC_STDCALL mfnmonitorset_(MFN *mfn,void (PETSC_STDCALL *monitor)(MFN*,PetscInt*,PetscReal*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)mfnmonitordefault_) {
    *ierr = MFNMonitorSet(*mfn,(PetscErrorCode (*)(MFN,PetscInt,PetscReal,void*))MFNMonitorDefault,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)mfnmonitorlg_) {
    *ierr = MFNMonitorSet(*mfn,MFNMonitorLG,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*mfn,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    *ierr = PetscObjectSetFortranCallback((PetscObject)*mfn,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
    *ierr = MFNMonitorSet(*mfn,ourmonitor,*mfn,ourdestroy);
  }
}

SLEPC_EXTERN void PETSC_STDCALL mfngettolerances_(MFN *mfn,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  CHKFORTRANNULLREAL(tol);
  CHKFORTRANNULLINTEGER(maxits);
  *ierr = MFNGetTolerances(*mfn,tol,maxits);
}

SLEPC_EXTERN void PETSC_STDCALL mfngettolerances00_(MFN *mfn,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  mfngettolerances_(mfn,tol,maxits,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL mfngettolerances10_(MFN *mfn,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  mfngettolerances_(mfn,tol,maxits,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL mfngettolerances01_(MFN *mfn,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  mfngettolerances_(mfn,tol,maxits,ierr);
}

