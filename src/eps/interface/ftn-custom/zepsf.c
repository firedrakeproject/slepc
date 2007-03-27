
#include "zpetsc.h"
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
#define epsmonitordefault_          EPSMONITORDEFAULT
#define epsmonitorlg_               EPSMONITORLG
#define epsmonitorset_              EPSMONITORSET
#define epsgetst_                   EPSGETST
#define epsgetwhicheigenpairs_      EPSGETWHICHEIGENPAIRS
#define epsgetproblemtype_          EPSGETPROBLEMTYPE
#define epsgetclass_                EPSGETCLASS
#define epsgetconvergedreason_      EPSGETCONVERGEDREASON
#define epsgetorthogonalization_    EPSGETORTHOGONALIZATION
#define epspowergetshifttype_       EPSPOWERGETSHIFTTYPE
#define epslanczosgetreorthog_      EPSLANCZOSGETREORTHOG
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsview_                    epsview
#define epssetoptionsprefix_        epssetoptionsprefix
#define epsappendoptionsprefix_     epsappendoptionsprefix
#define epsgetoptionsprefix_        epsgetoptionsprefix
#define epscreate_                  epscreate
#define epssettype_                 epssettype           
#define epsgettype_                 epsgettype
#define epsmonitordefault_          epsmonitordefault
#define epsmonitorlg_               epsmonitorlg
#define epsmonitorset_              epsmonitorset
#define epsgetst_                   epsgetst
#define epsgetwhicheigenpairs_      epsgetwhicheigenpairs
#define epsgetproblemtype_          epsgetproblemtype
#define epsgetclass_                epsgetclass
#define epsgetconvergedreason_      epsgetconvergedreason
#define epsgetorthogonalization_    epsgetorthogonalization
#define epspowergetshifttype_       epspowergetshifttype
#define epslanczosgetreorthog_      epslanczosgetreorthog
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f1)(EPS*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,void*,PetscErrorCode*);
static void (PETSC_STDCALL *f2)(void*,PetscErrorCode*);

/*
   These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code, hence no STDCALL
*/
void epsmonitordefault_(EPS *eps,int *it,int *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorDefault(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}

void epsmonitorlg_(EPS *eps,int *it,int *nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int *nest,void *ctx,PetscErrorCode *ierr)
{
  *ierr = EPSMonitorLG(*eps,*it,*nconv,eigr,eigi,errest,*nest,ctx);
}
EXTERN_C_END
 
/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmonitor(EPS eps,int i,int nc,PetscScalar *er,PetscScalar *ei,PetscReal *d,int l,void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f1)(&eps,&i,&nc,er,ei,d,&l,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourdestroy(void* ctx)
{
  PetscErrorCode ierr = 0;
  (*f2)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

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
  const char *tname;

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

void PETSC_STDCALL epssetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSSetOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epsappendoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSAppendOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epscreate_(MPI_Fint *comm,EPS *eps,PetscErrorCode *ierr)
{
  *ierr = EPSCreate(MPI_Comm_f2c(*(comm)),eps);
}

void PETSC_STDCALL epsmonitorset_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,int*,int*,PetscScalar*,PetscScalar*,PetscReal*,int*,void*,PetscErrorCode*),
                                  void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,PetscErrorCode *),PetscErrorCode *ierr)
{
  if ((void(*)())monitor == (void(*)())epsmonitordefault_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorDefault,0,0);
  } else if ((void(*)())monitor == (void(*)())epsmonitorlg_) {
    *ierr = EPSMonitorSet(*eps,EPSMonitorLG,0,0);
  } else {
    f1  = monitor;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = EPSMonitorSet(*eps,ourmonitor,mctx,0);
    } else {
      f2 = monitordestroy;
      *ierr = EPSMonitorSet(*eps,ourmonitor,mctx,ourdestroy);
    }
  }
}

void PETSC_STDCALL epsgetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

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

void PETSC_STDCALL epsgetclass_(EPS *eps,EPSClass *cl,int *ierr)
{
  *ierr = EPSGetClass(*eps,cl);
}

void PETSC_STDCALL epsgetconvergedreason_(EPS *eps,EPSConvergedReason *reason,int *ierr)
{
  *ierr = EPSGetConvergedReason(*eps,reason);
}

void PETSC_STDCALL epsgetorthogonalization_(EPS *eps,EPSOrthogonalizationType *type,EPSOrthogonalizationRefinementType *refinement,PetscReal *eta,int *ierr)
{
  *ierr = EPSGetOrthogonalization(*eps,type,refinement,eta);
}

void PETSC_STDCALL epspowergetshifttype_(EPS *eps,EPSPowerShiftType *shift,int *ierr)
{
  *ierr = EPSPowerGetShiftType(*eps,shift);
}

void PETSC_STDCALL epslanczosgetreorthog_(EPS *eps,EPSLanczosReorthogType *reorthog,int *ierr)
{
  *ierr = EPSLanczosGetReorthog(*eps,reorthog);
}
EXTERN_C_END

