/*
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

#include <petsc/private/fortranimpl.h>
#include <slepc/private/slepcimpl.h>
#include <slepc/private/epsimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define epsview_                          EPSVIEW
#define epserrorview_                     EPSERRORVIEW
#define epsreasonview_                    EPSREASONVIEW
#define epsvaluesview_                    EPSVALUESVIEW
#define epsvectorsview_                   EPSVECTORSVIEW
#define epssetoptionsprefix_              EPSSETOPTIONSPREFIX
#define epsappendoptionsprefix_           EPSAPPENDOPTIONSPREFIX
#define epsgetoptionsprefix_              EPSGETOPTIONSPREFIX
#define epssettype_                       EPSSETTYPE
#define epsgettype_                       EPSGETTYPE
#define epsmonitorall_                    EPSMONITORALL
#define epsmonitorlg_                     EPSMONITORLG
#define epsmonitorlgall_                  EPSMONITORLGALL
#define epsmonitorset_                    EPSMONITORSET
#define epsmonitorconverged_              EPSMONITORCONVERGED
#define epsmonitorfirst_                  EPSMONITORFIRST
#define epsconvergedabsolute_             EPSCONVERGEDABSOLUTE
#define epsconvergedrelative_             EPSCONVERGEDRELATIVE
#define epsconvergednorm_                 EPSCONVERGEDNORM
#define epssetconvergencetestfunction_    EPSSETCONVERGENCETESTFUNCTION
#define epssetstoppingtestfunction_       EPSSETSTOPPINGTESTFUNCTION
#define epsseteigenvaluecomparison_       EPSSETEIGENVALUECOMPARISON
#define epssetarbitraryselection_         EPSSETARBITRARYSELECTION
#define epskrylovschursetsubintervals_    EPSKRYLOVSCHURSETSUBINTERVALS
#define epskrylovschurgetsubintervals_    EPSKRYLOVSCHURGETSUBINTERVALS
#define epskrylovschurgetsubcomminfo_     EPSKRYLOVSCHURGETSUBCOMMINFO
#define epskrylovschurgetsubcommpairs_    EPSKRYLOVSCHURGETSUBCOMMPAIRS
#define epskrylovschurgetsubcommmats_     EPSKRYLOVSCHURGETSUBCOMMMATS
#define epskrylovschurupdatesubcommmats_  EPSKRYLOVSCHURUPDATESUBCOMMMATS
#define epsgetdimensions000_              EPSGETDIMENSIONS000
#define epsgetdimensions100_              EPSGETDIMENSIONS100
#define epsgetdimensions010_              EPSGETDIMENSIONS010
#define epsgetdimensions001_              EPSGETDIMENSIONS001
#define epsgetdimensions110_              EPSGETDIMENSIONS110
#define epsgetdimensions011_              EPSGETDIMENSIONS011
#define epsgetdimensions101_              EPSGETDIMENSIONS101
#define epssetoperators_                  EPSSETOPERATORS
#define epsgeteigenpair00_                EPSGETEIGENPAIR00
#define epsgeteigenpair10_                EPSGETEIGENPAIR10
#define epsgeteigenpair01_                EPSGETEIGENPAIR01
#define epsgeteigenpair11_                EPSGETEIGENPAIR11
#define epsgeteigenvalue00_               EPSGETEIGENVALUE00
#define epsgeteigenvalue10_               EPSGETEIGENVALUE10
#define epsgeteigenvalue01_               EPSGETEIGENVALUE01
#define epsgeteigenvector_                EPSGETEIGENVECTOR
#define epsgettolerances00_               EPSGETTOLERANCES00
#define epsgettolerances10_               EPSGETTOLERANCES10
#define epsgettolerances01_               EPSGETTOLERANCES01
#define epsgetbalance000_                 EPSGETBALANCE000
#define epsgetbalance100_                 EPSGETBALANCE100
#define epsgetbalance010_                 EPSGETBALANCE010
#define epsgetbalance001_                 EPSGETBALANCE001
#define epsgetbalance110_                 EPSGETBALANCE110
#define epsgetbalance011_                 EPSGETBALANCE011
#define epsgetbalance101_                 EPSGETBALANCE101
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsview_                          epsview
#define epserrorview_                     epserrorview
#define epsreasonview_                    epsreasonview
#define epsvaluesview_                    epsvaluesview
#define epsvectorsview_                   epsvectorsview
#define epssetoptionsprefix_              epssetoptionsprefix
#define epsappendoptionsprefix_           epsappendoptionsprefix
#define epsgetoptionsprefix_              epsgetoptionsprefix
#define epssettype_                       epssettype
#define epsgettype_                       epsgettype
#define epsmonitorall_                    epsmonitorall
#define epsmonitorlg_                     epsmonitorlg
#define epsmonitorlgall_                  epsmonitorlgall
#define epsmonitorset_                    epsmonitorset
#define epsmonitorconverged_              epsmonitorconverged
#define epsmonitorfirst_                  epsmonitorfirst
#define epsconvergedabsolute_             epsconvergedabsolute
#define epsconvergedrelative_             epsconvergedrelative
#define epsconvergednorm_                 epsconvergednorm
#define epssetconvergencetestfunction_    epssetconvergencetestfunction
#define epssetstoppingtestfunction_       epssetstoppingtestfunction
#define epsseteigenvaluecomparison_       epsseteigenvaluecomparison
#define epssetarbitraryselection_         epssetarbitraryselection
#define epskrylovschursetsubintervals_    epskrylovschursetsubintervals
#define epskrylovschurgetsubintervals_    epskrylovschurgetsubintervals
#define epskrylovschurgetsubcomminfo_     epskrylovschurgetsubcomminfo
#define epskrylovschurgetsubcommpairs_    epskrylovschurgetsubcommpairs
#define epskrylovschurgetsubcommmats_     epskrylovschurgetsubcommmats
#define epskrylovschurupdatesubcommmats_  epskrylovschurupdatesubcommmats
#define epsgetdimensions000_              epsgetdimensions000
#define epsgetdimensions100_              epsgetdimensions100
#define epsgetdimensions010_              epsgetdimensions010
#define epsgetdimensions001_              epsgetdimensions001
#define epsgetdimensions110_              epsgetdimensions110
#define epsgetdimensions011_              epsgetdimensions011
#define epsgetdimensions101_              epsgetdimensions101
#define epssetoperators_                  epssetoperators
#define epsgeteigenpair00_                epsgeteigenpair00
#define epsgeteigenpair10_                epsgeteigenpair10
#define epsgeteigenpair01_                epsgeteigenpair01
#define epsgeteigenpair11_                epsgeteigenpair11
#define epsgeteigenvalue00_               epsgeteigenvalue00
#define epsgeteigenvalue10_               epsgeteigenvalue10
#define epsgeteigenvalue01_               epsgeteigenvalue01
#define epsgeteigenvector_                epsgeteigenvector
#define epsgettolerances00_               epsgettolerances00
#define epsgettolerances10_               epsgettolerances10
#define epsgettolerances01_               epsgettolerances01
#define epsgetbalance000_                 epsgetbalance000
#define epsgetbalance100_                 epsgetbalance100
#define epsgetbalance010_                 epsgetbalance010
#define epsgetbalance001_                 epsgetbalance001
#define epsgetbalance110_                 epsgetbalance110
#define epsgetbalance011_                 epsgetbalance011
#define epsgetbalance101_                 epsgetbalance101
#endif

/*
   These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code, hence no STDCALL
*/
PETSC_EXTERN void epsmonitorall_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorAll(*eps,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void epsmonitorconverged_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,SlepcConvMonitor *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorConverged(*eps,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void epsmonitorfirst_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,PetscViewerAndFormat **ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorFirst(*eps,*it,*nconv,eigr,eigi,errest,*nest,*ctx);
}

PETSC_EXTERN void epsmonitorlg_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorLG(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

PETSC_EXTERN void epsmonitorlgall_(EPS *eps,PetscInt *it,PetscInt *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorLGAll(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

static struct {
  PetscFortranCallbackId monitor;
  PetscFortranCallbackId monitordestroy;
  PetscFortranCallbackId convergence;
  PetscFortranCallbackId convdestroy;
  PetscFortranCallbackId stopping;
  PetscFortranCallbackId stopdestroy;
  PetscFortranCallbackId comparison;
  PetscFortranCallbackId arbitrary;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmonitor(EPS eps,PetscInt i,PetscInt nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,PetscInt l,void* ctx)
{
  PetscObjectUseFortranCallback(eps,_cb.monitor,(EPS*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),(&eps,&i,&nc,er,ei,d,&l,_ctx,&ierr));
}

static PetscErrorCode ourdestroy(void** ctx)
{
  EPS eps = (EPS)*ctx;
  PetscObjectUseFortranCallback(eps,_cb.monitordestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

static PetscErrorCode ourconvergence(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscObjectUseFortranCallback(eps,_cb.convergence,(EPS*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),(&eps,&eigr,&eigi,&res,errest,_ctx,&ierr));
}

static PetscErrorCode ourconvdestroy(void *ctx)
{
  EPS eps = (EPS)ctx;
  PetscObjectUseFortranCallback(eps,_cb.convdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

static PetscErrorCode ourstopping(EPS eps,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,EPSConvergedReason *reason,void *ctx)
{
  PetscObjectUseFortranCallback(eps,_cb.stopping,(EPS*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,EPSConvergedReason*,void*,PetscErrorCode*),(&eps,&its,&max_it,&nconv,&nev,reason,_ctx,&ierr));
}

static PetscErrorCode ourstopdestroy(void *ctx)
{
  EPS eps = (EPS)ctx;
  PetscObjectUseFortranCallback(eps,_cb.stopdestroy,(void*,PetscErrorCode*),(_ctx,&ierr));
}

static PetscErrorCode oureigenvaluecomparison(PetscScalar ar,PetscScalar ai,PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  EPS eps = (EPS)ctx;
  PetscObjectUseFortranCallback(eps,_cb.comparison,(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*,PetscErrorCode*),(&ar,&ai,&br,&bi,r,_ctx,&ierr));
}

static PetscErrorCode ourarbitraryfunc(PetscScalar er,PetscScalar ei,Vec xr,Vec xi,PetscScalar *rr,PetscScalar *ri,void *ctx)
{
  EPS eps = (EPS)ctx;
  PetscObjectUseFortranCallback(eps,_cb.arbitrary,(PetscScalar*,PetscScalar*,Vec*,Vec*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*),(&er,&ei,&xr,&xi,rr,ri,_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL epsview_(EPS *eps,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSView(*eps,v);
}

PETSC_EXTERN void PETSC_STDCALL epsreasonview_(EPS *eps,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSReasonView(*eps,v);
}

PETSC_EXTERN void PETSC_STDCALL epserrorview_(EPS *eps,EPSErrorType *etype,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSErrorView(*eps,*etype,v);
}

PETSC_EXTERN void PETSC_STDCALL epsvaluesview_(EPS *eps,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSValuesView(*eps,v);
}

PETSC_EXTERN void PETSC_STDCALL epsvectorsview_(EPS *eps,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSVectorsView(*eps,v);
}

PETSC_EXTERN void PETSC_STDCALL epssettype_(EPS *eps,char *type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = EPSSetType(*eps,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL epsgettype_(EPS *eps,char *name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  EPSType tname;

  *ierr = EPSGetType(*eps,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL epssetoptionsprefix_(EPS *eps,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSSetOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL epsappendoptionsprefix_(EPS *eps,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSAppendOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL epsgetoptionsprefix_(EPS *eps,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = EPSGetOptionsPrefix(*eps,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL epsmonitorset_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*,void*,PetscErrorCode*),void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mctx);
  CHKFORTRANNULLFUNCTION(monitordestroy);
  if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorall_) {
    *ierr = EPSMonitorSet(*eps,(PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))EPSMonitorAll,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorconverged_) {
    *ierr = EPSMonitorSet(*eps,(PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))EPSMonitorConverged,*(SlepcConvMonitor*)mctx,(PetscErrorCode (*)(void**))SlepcConvMonitorDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorfirst_) {
    *ierr = EPSMonitorSet(*eps,(PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*))EPSMonitorFirst,*(PetscViewerAndFormat**)mctx,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlg_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLG,0,0);
  } else if ((PetscVoidFunction)monitor == (PetscVoidFunction)epsmonitorlgall_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLGAll,0,0);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitor,(PetscVoidFunction)monitor,mctx); if (*ierr) return;
    if (!monitordestroy) {
      *ierr = EPSMonitorSet(*eps,ourmonitor,*eps,0);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.monitordestroy,(PetscVoidFunction)monitordestroy,mctx); if (*ierr) return;
      *ierr = EPSMonitorSet(*eps,ourmonitor,*eps,ourdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL epsconvergedabsolute_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSConvergedAbsolute(*eps,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL epsconvergedrelative_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSConvergedRelative(*eps,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL epsconvergednorm_(EPS *eps,PetscScalar *eigr,PetscScalar *eigi,PetscReal *res,PetscReal *errest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSConvergedNorm(*eps,*eigr,*eigi,*res,errest,ctx);
}

PETSC_EXTERN void PETSC_STDCALL epssetconvergencetestfunction_(EPS *eps,void (PETSC_STDCALL *func)(EPS*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(destroy);
  if ((PetscVoidFunction)func == (PetscVoidFunction)epsconvergedabsolute_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_ABS);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)epsconvergedrelative_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_REL);
  } else if ((PetscVoidFunction)func == (PetscVoidFunction)epsconvergednorm_) {
    *ierr = EPSSetConvergenceTest(*eps,EPS_CONV_NORM);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convergence,(PetscVoidFunction)func,ctx); if (*ierr) return;
    if (!destroy) {
      *ierr = EPSSetConvergenceTestFunction(*eps,ourconvergence,*eps,NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.convdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
      *ierr = EPSSetConvergenceTestFunction(*eps,ourconvergence,*eps,ourconvdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL epsstoppingbasic_(EPS *eps,PetscInt *its,PetscInt *max_it,PetscInt *nconv,PetscInt *nev,EPSConvergedReason *reason,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSStoppingBasic(*eps,*its,*max_it,*nconv,*nev,reason,ctx);
}

PETSC_EXTERN void PETSC_STDCALL epssetstoppingtestfunction_(EPS *eps,void (PETSC_STDCALL *func)(EPS*,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*,PetscErrorCode*),void* ctx,void (PETSC_STDCALL *destroy)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  CHKFORTRANNULLFUNCTION(destroy);
  if ((PetscVoidFunction)func == (PetscVoidFunction)epsstoppingbasic_) {
    *ierr = EPSSetStoppingTest(*eps,EPS_STOP_BASIC);
  } else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopping,(PetscVoidFunction)func,ctx); if (*ierr) return;
    if (!destroy) {
      *ierr = EPSSetStoppingTestFunction(*eps,ourstopping,*eps,NULL);
    } else {
      *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.stopdestroy,(PetscVoidFunction)destroy,ctx); if (*ierr) return;
      *ierr = EPSSetStoppingTestFunction(*eps,ourstopping,*eps,ourstopdestroy);
    }
  }
}

PETSC_EXTERN void PETSC_STDCALL epsseteigenvaluecomparison_(EPS *eps,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,void*),void* ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.comparison,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = EPSSetEigenvalueComparison(*eps,oureigenvaluecomparison,*eps);
}

PETSC_EXTERN void PETSC_STDCALL epssetarbitraryselection_(EPS *eps,void (PETSC_STDCALL *func)(PetscScalar*,PetscScalar*,Vec*,Vec*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*eps,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.arbitrary,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = EPSSetArbitrarySelection(*eps,ourarbitraryfunc,*eps);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschursetsubintervals_(EPS *eps,PetscReal *subint,PetscErrorCode *ierr)
{
  CHKFORTRANNULLREAL(subint);
  *ierr = EPSKrylovSchurSetSubintervals(*eps,subint);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubintervals_(EPS *eps,PetscReal *subint,PetscErrorCode *ierr)
{
  PetscReal *osubint;
  PetscInt  npart;

  CHKFORTRANNULLREAL(subint);
  *ierr = EPSKrylovSchurGetSubintervals(*eps,&osubint); if (*ierr) return;
  *ierr = EPSKrylovSchurGetPartitions(*eps,&npart); if (*ierr) return;
  *ierr = PetscMemcpy(subint,osubint,(npart+1)*sizeof(PetscReal)); if (*ierr) return;
  *ierr = PetscFree(osubint);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubcomminfo_(EPS *eps,PetscInt *k,PetscInt *n,Vec *v,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(v);
  *ierr = EPSKrylovSchurGetSubcommInfo(*eps,k,n,v);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubcommpairs_(EPS *eps,PetscInt *i,PetscScalar *eig,Vec *v,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(v);
  *ierr = EPSKrylovSchurGetSubcommPairs(*eps,*i,eig,*v);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubcommmats_(EPS *eps,Mat *A,Mat *B,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(A);
  CHKFORTRANNULLOBJECT(B);
  *ierr = EPSKrylovSchurGetSubcommMats(*eps,A,B);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurupdatesubcommmats_(EPS *eps,PetscScalar *s,PetscScalar *a,Mat *Au,PetscScalar *t,PetscScalar *b,Mat *Bu,MatStructure *str,PetscBool *globalup,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(Au);
  CHKFORTRANNULLOBJECTDEREFERENCE(Bu);
  *ierr = EPSKrylovSchurUpdateSubcommMats(*eps,*s,*a,*Au,*t,*b,*Bu,*str,*globalup);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  CHKFORTRANNULLINTEGER(nev);
  CHKFORTRANNULLINTEGER(ncv);
  CHKFORTRANNULLINTEGER(mpd);  
  *ierr = EPSGetDimensions(*eps,nev,ncv,mpd);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions000_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions100_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions010_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions001_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions110_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions011_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetdimensions101_(EPS *eps,PetscInt *nev,PetscInt *ncv,PetscInt *mpd,int *ierr)
{
  epsgetdimensions_(eps,nev,ncv,mpd,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epssetoperators_(EPS *eps,Mat *A,Mat *B,int *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(A);
  CHKFORTRANNULLOBJECTDEREFERENCE(B);  
  *ierr = EPSSetOperators(*eps,*A,*B);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenpair_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  CHKFORTRANNULLSCALAR(eigr);
  CHKFORTRANNULLSCALAR(eigi);
  CHKFORTRANNULLOBJECTDEREFERENCE(Vr);
  CHKFORTRANNULLOBJECTDEREFERENCE(Vi);
  *ierr = EPSGetEigenpair(*eps,*i,eigr,eigi,*Vr,*Vi);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenpair00_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  epsgeteigenpair_(eps,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenpair10_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  epsgeteigenpair_(eps,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenpair01_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  epsgeteigenpair_(eps,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenpair11_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,Vec *Vr,Vec *Vi,int *ierr)
{
  epsgeteigenpair_(eps,i,eigr,eigi,Vr,Vi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenvalue_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,int *ierr)
{
  CHKFORTRANNULLSCALAR(eigr);
  CHKFORTRANNULLSCALAR(eigi);
  *ierr = EPSGetEigenvalue(*eps,*i,eigr,eigi);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenvalue00_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,int *ierr)
{
  epsgeteigenvalue_(eps,i,eigr,eigi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenvalue10_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,int *ierr)
{
  epsgeteigenvalue_(eps,i,eigr,eigi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenvalue01_(EPS *eps,PetscInt *i,PetscScalar *eigr,PetscScalar *eigi,int *ierr)
{
  epsgeteigenvalue_(eps,i,eigr,eigi,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgeteigenvector_(EPS *eps,PetscInt *i,Vec *Vr,Vec *Vi,int *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(Vr);
  CHKFORTRANNULLOBJECTDEREFERENCE(Vi);
  *ierr = EPSGetEigenvector(*eps,*i,*Vr,*Vi);
}

PETSC_EXTERN void PETSC_STDCALL epsgettolerances_(EPS *eps,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  CHKFORTRANNULLREAL(tol);
  CHKFORTRANNULLINTEGER(maxits);
  *ierr = EPSGetTolerances(*eps,tol,maxits);
}

PETSC_EXTERN void PETSC_STDCALL epsgettolerances00_(EPS *eps,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  epsgettolerances_(eps,tol,maxits,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgettolerances10_(EPS *eps,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  epsgettolerances_(eps,tol,maxits,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgettolerances01_(EPS *eps,PetscReal *tol,PetscInt *maxits,int *ierr)
{
  epsgettolerances_(eps,tol,maxits,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  CHKFORTRANNULLINTEGER(bal);
  CHKFORTRANNULLINTEGER(its);
  CHKFORTRANNULLREAL(cutoff);  
  *ierr = EPSGetBalance(*eps,bal,its,cutoff);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance000_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance100_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance010_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance001_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance110_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance011_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

PETSC_EXTERN void PETSC_STDCALL epsgetbalance101_(EPS *eps,EPSBalance *bal,PetscInt *its,PetscReal *cutoff,int *ierr)
{
  epsgetbalance_(eps,bal,its,cutoff,ierr);
}

