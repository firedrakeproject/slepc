/*
   Modification of the *temp* implementation of the BLOPEX multivector in order
   to wrap created PETSc vectors as multivectors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "petscsys.h"
#include "petscvec.h"
#include "petscmat.h"
#include <assert.h>
#include "petscblaslapack.h"
#include "interpreter.h"
#include "temp_multivector.h"
#include "../src/contrib/blopex/petsc-interface/petsc-interface.h"
#include "slepc-interface.h"

static void* mv_TempMultiVectorCreateFromPETScVector( void* ii_, int n, void* sample )
{
  int i;
  Vec *vecs = (Vec*)sample;

  mv_TempMultiVector* x;
  mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;

  x = (mv_TempMultiVector*) malloc(sizeof(mv_TempMultiVector));
  assert( x != NULL );
  
  x->interpreter = ii;
  x->numVectors = n;
  
  x->vector = (void**) calloc( n, sizeof(void*) );
  assert( x->vector != NULL );

  x->ownsVectors = 1;
  x->mask = NULL;
  x->ownsMask = 0;

  for ( i = 0; i < n; i++ ) {
    x->vector[i] = (void*)vecs[i];
  }

  return x;
}

static void mv_TempMultiPETSCVectorDestroy( void* x_ )
{
  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  if ( x == NULL )
    return;

  if ( x->ownsVectors && x->vector != NULL ) {
       free(x->vector);
  }
  if ( x->mask && x->ownsMask )
    free(x->mask);
  free(x);
}


/*
    Create an InterfaceInterpreter using the PETSc implementation
    but with a modified CreateMultiVector that doesn't create any
    new vector.
*/
int SLEPCSetupInterpreter( mv_InterfaceInterpreter *i )
{

  i->CreateVector = PETSC_MimicVector;
  i->DestroyVector = PETSC_DestroyVector;
  i->InnerProd = PETSC_InnerProd;
  i->CopyVector = PETSC_CopyVector;
  i->ClearVector = PETSC_ClearVector;
  i->SetRandomValues = PETSC_SetRandomValues;
  i->ScaleVector = PETSC_ScaleVector;
  i->Axpy = PETSC_Axpy;

  /* Multivector part */

  i->CreateMultiVector = mv_TempMultiVectorCreateFromPETScVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;

  return 0;
}

/*
    Change the multivector destructor in order to destroy the multivector
    structure without destroy the PETSc vectors.
*/
void SLEPCSetupInterpreterForDignifiedDeath( mv_InterfaceInterpreter *i )
{
  i->DestroyMultiVector = mv_TempMultiPETSCVectorDestroy;
}

