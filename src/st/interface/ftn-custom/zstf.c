
#include "zpetsc.h"
#include "slepcst.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define stsettype_                STSETTYPE           
#define stgettype_                STGETTYPE
#define stdestroy_                STDESTROY
#define stcreate_                 STCREATE
#define stgetoperators_           STGETOPERATORS
#define stsetoptionsprefix_       STSETOPTIONSPREFIX
#define stappendoptionsprefix_    STAPPENDOPTIONSPREFIX
#define stgetoptionsprefix_       STGETOPTIONSPREFIX
#define stview_                   STVIEW
#define stgetmatmode_             STGETMATMODE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define stsettype_                stsettype
#define stgettype_                stgettype
#define stdestroy_                stdestroy
#define stcreate_                 stcreate
#define stgetoperators_           stgetoperators
#define stsetoptionsprefix_       stsetoptionsprefix
#define stappendoptionsprefix_    stappendoptionsprefix
#define stgetoptionsprefix_       stgetoptionsprefix
#define stview_                   stview
#define stgetmatmode_             stgetmatmode
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL stsettype_(ST *st,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = STSetType(*st,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL stgettype_(ST *st,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = STGetType(*st,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
#endif
  FIXRETURNCHAR(name,len);
}

void PETSC_STDCALL stdestroy_(ST *st,PetscErrorCode *ierr)
{
  *ierr = STDestroy(*st);
}

void PETSC_STDCALL stcreate_(MPI_Comm *comm,ST *newst,PetscErrorCode *ierr)
{
  *ierr = STCreate((MPI_Comm)PetscToPointerComm(*comm),newst);
}

void PETSC_STDCALL stgetoperators_(ST *st,Mat *mat,Mat *pmat,PetscErrorCode *ierr)
{
  if (FORTRANNULLOBJECT(mat))   mat = PETSC_NULL;
  if (FORTRANNULLOBJECT(pmat))  pmat = PETSC_NULL;
  *ierr = STGetOperators(*st,mat,pmat);
}

void PETSC_STDCALL stsetoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STSetOptionsPrefix(*st,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL stappendoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),
                                          PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STAppendOptionsPrefix(*st,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL stgetoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = STGetOptionsPrefix(*st,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
#endif
  FIXRETURNCHAR(prefix,len);
}

void PETSC_STDCALL stview_(ST *st,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = STView(*st,v);
}

void PETSC_STDCALL  stgetmatmode_(ST *st,STMatMode *mode,int *ierr)
{
  *ierr = STGetMatMode(*st,mode);
}

EXTERN_C_END

