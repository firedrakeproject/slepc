
#ifndef _STIMPL
#define _STIMPL

#include "slepceps.h"

typedef struct _STOps *STOps;

struct _STOps {
  int          (*setup)(ST);
  int          (*apply)(ST,Vec,Vec);
  int          (*applyB)(ST,Vec,Vec);
  int          (*applynoB)(ST,Vec,Vec);
  int          (*setshift)(ST,PetscScalar);
  int          (*setfromoptions)(ST);
  int          (*presolve)(ST);  
  int          (*postsolve)(ST);  
  int          (*backtr)(ST,PetscScalar*,PetscScalar*);  
  int          (*destroy)(ST);
  int          (*view)(ST,PetscViewer);
};

struct _p_ST {
  PETSCHEADER(struct _STOps)
  /*------------------------- User parameters --------------------------*/
  Mat          A,B;              /* Matrices which define the eigensystem */
  PetscScalar  sigma;            /* Value of the shift */

  /*------------------------- Misc data --------------------------*/
  Vec          vec;
  KSP          ksp;
  void         *data;
  int          setupcalled;
  int          numberofshifts;
  int          lineariterations;
  int          (*checknullspace)(ST,int,Vec*);
};

extern int STDefaultApplyB( ST, Vec, Vec );
extern int STAssociatedKSPSolve(ST,Vec,Vec);
extern int STCheckNullSpace_Default(ST,int,Vec*);

#endif

