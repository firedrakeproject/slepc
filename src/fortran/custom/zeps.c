
#include "src/fortran/custom/zpetsc.h"
#include "slepceps.h"
#include "src/eps/epsimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define epsview_                    EPSVIEW
#define epssetoptionsprefix_        EPSSETOPTIONSPREFIX
#define epsappendoptionsprefix_     EPSAPPENDOPTIONSPREFIX
#define epsgetoptionsprefix_        EPSGETOPTIONSPREFIX
#define epscreate_                  EPSCREATE
#define epssettype_                 EPSSETTYPE           
#define epsgettype_                 EPSGETTYPE
#define epsdefaultestimatesmonitor_ EPSDEFAULTESTIMATESMONITOR
#define epsdefaultvaluesmonitor_    EPSDEFAULTVALUESMONITOR
#define epssetmonitor_              EPSSETMONITOR
#define epssetvaluesmonitor_        EPSSETVALUESMONITOR
#define epssetst_                   EPSSETST
#define epsgetwhicheigenpairs_      EPSGETWHICHEIGENPAIRS
#define epsgetproblemtype_          EPSGETPROBLEMTYPE
#define epsgetorthogonalization_    EPSGETORTHOGONALIZATION
#define epspowergetshifttype_       EPSPOWERGETSHIFTTYPE
#define epslanczosgetorthog_        EPSLANCZOSGETORTHOG
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsview_                    epsview
#define epssetoptionsprefix_        epssetoptionsprefix
#define epsappendoptionsprefix_     epsappendoptionsprefix
#define epsgetoptionsprefix_        epsgetoptionsprefix
#define epscreate_                  epscreate
#define epssettype_                 epssettype           
#define epsgettype_                 epsgettype
#define epsdefaultestimatesmonitor_ epsdefaultestimatesmonitor
#define epsdefaultvaluesmonitor_    epsdefaultvaluesmonitor
#define epssetmonitor_              epssetmonitor
#define epssetvaluesmonitor_        epssetvaluesmonitor
#define epssetst_                   epssetst
#define epsgetwhicheigenpairs_      epsgetwhicheigenpairs
#define epsgetproblemtype_          epsgetproblemtype
#define epsgetorthogonalization_    epsgetorthogonalization
#define epspowergetshifttype_       epspowergetshifttype
#define epslanczosgetorthog_        epslanczosgetorthog
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL epsview_(EPS *eps,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSView(*eps,v);
}

void PETSC_STDCALL epssettype_(EPS *eps,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = EPSSetType(*eps,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL epsgettype_(EPS *eps,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = EPSGetType(*eps,&tname);if (*ierr) return;
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1); 
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
  FIXRETURNCHAR(name,len);
}

void PETSC_STDCALL epssetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),
                                        PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSSetOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epsappendoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSAppendOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epscreate_(MPI_Comm *comm,EPS *eps,PetscErrorCode *ierr){
  *ierr = EPSCreate((MPI_Comm)PetscToPointerComm(*comm),eps);
}

/*
        These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code
   
   functions, hence no STDCALL
*/
void  epsdefaultmonitor_(EPS *eps,int *it,int *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSDefaultMonitor(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}
 
static void (PETSC_STDCALL *f1)(EPS*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,void*,int*);
static int ourmonitor(EPS eps,int i,int nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,int l,void* ctx)
{
  int ierr = 0;
  (*f1)(&eps,&i,&nc,er,ei,d,&l,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL epssetmonitor_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,void*,int*),
                    void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,int *),PetscErrorCode *ierr)
{
  if ((void(*)())monitor == (void(*)())epsdefaultmonitor_) {
    *ierr = EPSSetMonitor(*eps,EPSDefaultMonitor,0);
  } else {
    f1  = monitor;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = EPSSetMonitor(*eps,ourmonitor,mctx);
    } else {
      *ierr = EPSSetMonitor(*eps,ourmonitor,mctx);
    }
  }
}

void PETSC_STDCALL epsgetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = EPSGetOptionsPrefix(*eps,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
#endif
  FIXRETURNCHAR(prefix,len);
}

void PETSC_STDCALL epsgetst_(EPS *eps,ST *st,int *ierr)
{
  *ierr = EPSGetST(*eps,st);
}

void PETSC_STDCALL epsgetwhicheigenpairs_(EPS *eps,EPSWhich *which,int *ierr)
{
  *ierr = EPSGetWhichEigenpairs(*eps,which);
}

void PETSC_STDCALL epsgetproblemtype_(EPS *eps,EPSProblemType *type,int *ierr)
{
  *ierr = EPSGetProblemType(*eps,type);
}

void PETSC_STDCALL epsgetorthogonalization_(EPS *eps,EPSOrthogonalizationType *type,EPSOrthogonalizationRefinementType *refinement,PetscReal *eta,int *ierr)
{
  *ierr = EPSGetOrthogonalization(*eps,type,refinement,eta);
}

void PETSC_STDCALL epspowergetshifttype_(EPS *eps,EPSPowerShiftType *shift,int *ierr)
{
  *ierr = EPSPowerGetShiftType(*eps,shift);
}

void PETSC_STDCALL epslanczosgetorthog_(EPS *eps,EPSLanczosOrthogType *reorthog,int *ierr)
{
  *ierr = EPSLanczosGetOrthog(*eps,reorthog);
}
EXTERN_C_END

