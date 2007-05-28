
#include "zpetsc.h"
#include "slepcip.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define ipcreate_                 IPCREATE
#define ipsetoptionsprefix_       IPSETOPTIONSPREFIX
#define ipappendoptionsprefix_    IPAPPENDOPTIONSPREFIX
#define ipgetorthogonalization_   IPGETORTHOGONALIZATION
#define ipview_                   IPVIEW
#define ipgetbilinarform_         IPGETBILINARFORM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ipcreate_                 ipcreate
#define ipsetoptionsprefix_       ipsetoptionsprefix
#define ipappendoptionsprefix_    ipappendoptionsprefix
#define ipgetorthogonalization_   ipgetorthogonalization
#define ipview_                   ipview
#define ipgetbilinarform_         ipgetbilinarform
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL ipcreate_(MPI_Fint *comm,IP *newip,PetscErrorCode *ierr)
{
  *ierr = IPCreate(MPI_Comm_f2c(*(comm)),newip);
}

void PETSC_STDCALL ipsetoptionsprefix_(IP *ip,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = IPSetOptionsPrefix(*ip,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL ipappendoptionsprefix_(IP *ip,CHAR prefix PETSC_MIXED_LEN(len),
                                          PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = IPAppendOptionsPrefix(*ip,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL ipgetorthogonalization_(IP *ip,IPOrthogonalizationType *type,IPOrthogonalizationRefinementType *refinement,PetscReal *eta,int *ierr)
{
  *ierr = IPGetOrthogonalization(*ip,type,refinement,eta);
}

void PETSC_STDCALL ipview_(IP *ip,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = IPView(*ip,v);
}

void PETSC_STDCALL ipgetbilinarform_(IP *ip,Mat *mat,IPBilinearForm* form,PetscErrorCode *ierr)
{
  if (FORTRANNULLOBJECT(mat)) mat = PETSC_NULL;  
  *ierr = IPGetBilinearForm(*ip,mat,form);
}

EXTERN_C_END
