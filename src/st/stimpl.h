
#ifndef _STIMPL
#define _STIMPL

#include "slepceps.h"

extern PetscEvent ST_SetUp, ST_Apply, ST_ApplyB, ST_ApplyNoB, ST_ApplyTranspose, ST_InnerProduct;
extern PetscFList STList;

typedef struct _STOps *STOps;

struct _STOps {
  int          (*setup)(ST);
  int          (*apply)(ST,Vec,Vec);
  int          (*applyB)(ST,Vec,Vec);
  int          (*applynoB)(ST,Vec,Vec);
  int          (*applytrans)(ST,Vec,Vec);
  int          (*setshift)(ST,PetscScalar);
  int          (*setfromoptions)(ST);
  int          (*presolve)(ST);  
  int          (*postsolve)(ST);  
  int          (*backtr)(ST,PetscScalar*,PetscScalar*);  
  int          (*destroy)(ST);
  int          (*view)(ST,PetscViewer);
};

struct _p_ST {
  PETSCHEADER(struct _STOps);
  /*------------------------- User parameters --------------------------*/
  Mat            A,B;              /* Matrices which define the eigensystem */
  PetscScalar    sigma;            /* Value of the shift */
  STMatMode      shift_matrix;
  STBilinearForm bilinear_form;
  MatStructure   str;          /* whether matrices have the same pattern or not */
  Mat            mat;

  /*------------------------- Misc data --------------------------*/
  KSP          ksp;
  Vec          w;
  void         *data;
  int          setupcalled;
  int          lineariterations;
  int          (*checknullspace)(ST,int,Vec*);
  
  /*------------------------- Cache Bx product -------------------*/
  PetscInt       xid;
  PetscInt       xstate;
  Vec            Bx;
};

EXTERN PetscErrorCode STRegisterAll(char*);
EXTERN PetscErrorCode STRegister(char*,char*,char*,int(*)(ST));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,0)
#else
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode STApplyB_Default(ST,Vec,Vec);
EXTERN PetscErrorCode STView_Default(ST,PetscViewer);
EXTERN PetscErrorCode STAssociatedKSPSolve(ST,Vec,Vec);
EXTERN PetscErrorCode STAssociatedKSPSolveTranspose(ST,Vec,Vec);
EXTERN PetscErrorCode STCheckNullSpace_Default(ST,int,Vec*);
EXTERN PetscErrorCode STMatShellCreate(ST st,Mat *mat);

#endif

