/*
   BV (basis vectors) interface routines, callable by users.

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

#include <slepc/private/bvimpl.h>            /*I "slepcbv.h" I*/

PetscClassId     BV_CLASSID = 0;
PetscLogEvent    BV_Create = 0,BV_Copy = 0,BV_Mult = 0,BV_MultVec = 0,BV_MultInPlace = 0,BV_Dot = 0,BV_DotVec = 0,BV_Orthogonalize = 0,BV_OrthogonalizeVec = 0,BV_Scale = 0,BV_Norm = 0,BV_NormVec = 0,BV_SetRandom = 0,BV_MatMult = 0,BV_MatMultVec = 0,BV_MatProject = 0;
static PetscBool BVPackageInitialized = PETSC_FALSE;

const char *BVOrthogTypes[] = {"CGS","MGS","BVOrthogType","BV_ORTHOG_",0};
const char *BVOrthogRefineTypes[] = {"IFNEEDED","NEVER","ALWAYS","BVOrthogRefineType","BV_ORTHOG_REFINE_",0};
const char *BVOrthogBlockTypes[] = {"GS","CHOL","BVOrthogBlockType","BV_ORTHOG_BLOCK_",0};
const char *BVMatMultTypes[] = {"VECS","MAT","MAT_SAVE","BVMatMultType","BV_MATMULT_",0};

#undef __FUNCT__
#define __FUNCT__ "BVFinalizePackage"
/*@C
   BVFinalizePackage - This function destroys everything in the Slepc interface
   to the BV package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode BVFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&BVList);CHKERRQ(ierr);
  BVPackageInitialized = PETSC_FALSE;
  BVRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVInitializePackage"
/*@C
   BVInitializePackage - This function initializes everything in the BV package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to BVCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode BVInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (BVPackageInitialized) PetscFunctionReturn(0);
  BVPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Basis Vectors",&BV_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = BVRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("BVCreate",BV_CLASSID,&BV_Create);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVCopy",BV_CLASSID,&BV_Copy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMult",BV_CLASSID,&BV_Mult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMultVec",BV_CLASSID,&BV_MultVec);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMultInPlace",BV_CLASSID,&BV_MultInPlace);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVDot",BV_CLASSID,&BV_Dot);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVDotVec",BV_CLASSID,&BV_DotVec);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVOrthogonalize",BV_CLASSID,&BV_Orthogonalize);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVOrthogonalizeV",BV_CLASSID,&BV_OrthogonalizeVec);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVScale",BV_CLASSID,&BV_Scale);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVNorm",BV_CLASSID,&BV_Norm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVNormVec",BV_CLASSID,&BV_NormVec);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVSetRandom",BV_CLASSID,&BV_SetRandom);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMatMult",BV_CLASSID,&BV_MatMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMatMultVec",BV_CLASSID,&BV_MatMultVec);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BVMatProject",BV_CLASSID,&BV_MatProject);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"bv",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(BV_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"bv",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(BV_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(BVFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy"
/*@
   BVDestroy - Destroys BV context that was created with BVCreate().

   Collective on BV

   Input Parameter:
.  bv - the basis vectors context

   Level: beginner

.seealso: BVCreate()
@*/
PetscErrorCode BVDestroy(BV *bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bv) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*bv,BV_CLASSID,1);
  if (--((PetscObject)(*bv))->refct > 0) { *bv = 0; PetscFunctionReturn(0); }
  if ((*bv)->ops->destroy) { ierr = (*(*bv)->ops->destroy)(*bv);CHKERRQ(ierr); }
  ierr = VecDestroy(&(*bv)->t);CHKERRQ(ierr);
  ierr = MatDestroy(&(*bv)->matrix);CHKERRQ(ierr);
  ierr = VecDestroy(&(*bv)->Bx);CHKERRQ(ierr);
  ierr = BVDestroy(&(*bv)->cached);CHKERRQ(ierr);
  ierr = PetscFree((*bv)->work);CHKERRQ(ierr);
  ierr = PetscFree2((*bv)->h,(*bv)->c);CHKERRQ(ierr);
  ierr = PetscFree((*bv)->omega);CHKERRQ(ierr);
  ierr = MatDestroy(&(*bv)->B);CHKERRQ(ierr);
  ierr = MatDestroy(&(*bv)->C);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*bv)->rand);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCreate"
/*@
   BVCreate - Creates a basis vectors context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  bv - location to put the basis vectors context

   Level: beginner

.seealso: BVSetUp(), BVDestroy(), BV
@*/
PetscErrorCode BVCreate(MPI_Comm comm,BV *newbv)
{
  PetscErrorCode ierr;
  BV             bv;

  PetscFunctionBegin;
  PetscValidPointer(newbv,2);
  *newbv = 0;
  ierr = BVInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(bv,BV_CLASSID,"BV","Basis Vectors","BV",comm,BVDestroy,BVView);CHKERRQ(ierr);

  bv->t            = NULL;
  bv->n            = -1;
  bv->N            = -1;
  bv->m            = 0;
  bv->l            = 0;
  bv->k            = 0;
  bv->nc           = 0;
  bv->orthog_type  = BV_ORTHOG_CGS;
  bv->orthog_ref   = BV_ORTHOG_REFINE_IFNEEDED;
  bv->orthog_eta   = 0.7071;
  bv->orthog_block = BV_ORTHOG_BLOCK_GS;
  bv->matrix       = NULL;
  bv->indef        = PETSC_FALSE;
  bv->vmm          = BV_MATMULT_MAT;

  bv->Bx           = NULL;
  bv->xid          = 0;
  bv->xstate       = 0;
  bv->cv[0]        = NULL;
  bv->cv[1]        = NULL;
  bv->ci[0]        = -1;
  bv->ci[1]        = -1;
  bv->st[0]        = -1;
  bv->st[1]        = -1;
  bv->id[0]        = 0;
  bv->id[1]        = 0;
  bv->h            = NULL;
  bv->c            = NULL;
  bv->omega        = NULL;
  bv->B            = NULL;
  bv->C            = NULL;
  bv->Aid          = 0;
  bv->defersfo     = PETSC_FALSE;
  bv->cached       = NULL;
  bv->bvstate      = 0;
  bv->rand         = NULL;
  bv->rrandom      = PETSC_FALSE;
  bv->work         = NULL;
  bv->lwork        = 0;
  bv->data         = NULL;

  *newbv = bv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVInsertVec"
/*@
   BVInsertVec - Insert a vector into the specified column.

   Collective on BV

   Input Parameters:
+  V - basis vectors
.  j - the column of V to be overwritten
-  w - the vector to be copied

   Level: intermediate

.seealso: BVInsertVecs()
@*/
PetscErrorCode BVInsertVec(BV V,PetscInt j,Vec w)
{
  PetscErrorCode ierr;
  PetscInt       n,N;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(V,j,2);
  PetscValidHeaderSpecific(w,VEC_CLASSID,3);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscCheckSameComm(V,1,w,3);

  ierr = VecGetSize(w,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(w,&n);CHKERRQ(ierr);
  if (N!=V->N || n!=V->n) SETERRQ4(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Vec sizes (global %D, local %D) do not match BV sizes (global %D, local %D)",N,n,V->N,V->n);
  if (j<-V->nc || j>=V->m) SETERRQ3(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %D, should be between %D and %D",j,-V->nc,V->m-1);

  ierr = BVGetColumn(V,j,&v);CHKERRQ(ierr);
  ierr = VecCopy(w,v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j,&v);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVInsertVecs"
/*@
   BVInsertVecs - Insert a set of vectors into the specified columns.

   Collective on BV

   Input Parameters:
+  V - basis vectors
.  s - first column of V to be overwritten
.  W - set of vectors to be copied
-  orth - flag indicating if the vectors must be orthogonalized

   Input/Output Parameter:
.  m - number of input vectors, on output the number of linearly independent
       vectors

   Notes:
   Copies the contents of vectors W to V(:,s:s+n). If the orthogonalization
   flag is set, then the vectors are copied one by one and then orthogonalized
   against the previous ones. If any of them is linearly dependent then it
   is discarded and the value of m is decreased.

   Level: intermediate

.seealso: BVInsertVec(), BVOrthogonalizeColumn()
@*/
PetscErrorCode BVInsertVecs(BV V,PetscInt s,PetscInt *m,Vec *W,PetscBool orth)
{
  PetscErrorCode ierr;
  PetscInt       n,N,i,ndep;
  PetscBool      lindep;
  PetscReal      norm;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(V,s,2);
  PetscValidPointer(m,3);
  PetscValidLogicalCollectiveInt(V,*m,3);
  if (!*m) PetscFunctionReturn(0);
  if (*m<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of vectors (given %D) cannot be negative",*m);
  PetscValidPointer(W,4);
  PetscValidHeaderSpecific(*W,VEC_CLASSID,4);
  PetscValidLogicalCollectiveBool(V,orth,5);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscCheckSameComm(V,1,*W,4);

  ierr = VecGetSize(*W,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(*W,&n);CHKERRQ(ierr);
  if (N!=V->N || n!=V->n) SETERRQ4(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Vec sizes (global %D, local %D) do not match BV sizes (global %D, local %D)",N,n,V->N,V->n);
  if (s<0 || s>=V->m) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument s has wrong value %D, should be between 0 and %D",s,V->m-1);
  if (s+(*m)>V->m) SETERRQ1(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Too many vectors provided, there is only room for %D",V->m);

  ndep = 0;
  for (i=0;i<*m;i++) {
    ierr = BVGetColumn(V,s+i-ndep,&v);CHKERRQ(ierr);
    ierr = VecCopy(W[i],v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,s+i-ndep,&v);CHKERRQ(ierr);
    if (orth) {
      ierr = BVOrthogonalizeColumn(V,s+i-ndep,NULL,&norm,&lindep);CHKERRQ(ierr);
      if (norm==0.0 || lindep) {
        ierr = PetscInfo1(V,"Removing linearly dependent vector %D\n",i);CHKERRQ(ierr);
        ndep++;
      } else {
        ierr = BVScaleColumn(V,s+i-ndep,1.0/norm);CHKERRQ(ierr);
      }
    }
  }
  *m -= ndep;
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVInsertConstraints"
/*@
   BVInsertConstraints - Insert a set of vectors as constraints.

   Collective on BV

   Input Parameters:
+  V - basis vectors
-  C - set of vectors to be inserted as constraints

   Input/Output Parameter:
.  nc - number of input vectors, on output the number of linearly independent
       vectors

   Notes:
   The constraints are relevant only during orthogonalization. Constraint
   vectors span a subspace that is deflated in every orthogonalization
   operation, so they are intended for removing those directions from the
   orthogonal basis computed in regular BV columns.

   Constraints are not stored in regular BV colums, but in a special part of
   the storage. They can be accessed with negative indices in BVGetColumn().

   This operation is DESTRUCTIVE, meaning that all data contained in the
   columns of V is lost. This is typically invoked just after creating the BV.
   Once a set of constraints has been set, it is not allowed to call this
   function again.

   The vectors are copied one by one and then orthogonalized against the
   previous ones. If any of them is linearly dependent then it is discarded
   and the value of nc is decreased. The behaviour is similar to BVInsertVecs().

   Level: advanced

.seealso: BVInsertVecs(), BVOrthogonalizeColumn(), BVGetColumn(), BVGetNumConstraints()
@*/
PetscErrorCode BVInsertConstraints(BV V,PetscInt *nc,Vec *C)
{
  PetscErrorCode ierr;
  PetscInt       msave;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidPointer(nc,2);
  PetscValidLogicalCollectiveInt(V,*nc,2);
  if (!*nc) PetscFunctionReturn(0);
  if (*nc<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of constraints (given %D) cannot be negative",*nc);
  PetscValidPointer(C,3);
  PetscValidHeaderSpecific(*C,VEC_CLASSID,3);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscCheckSameComm(V,1,*C,3);
  if (V->nc) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONGSTATE,"Constraints already present in this BV object");
  if (V->ci[0]!=-1 || V->ci[1]!=-1) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Cannot call BVInsertConstraints after BVGetColumn");

  msave = V->m;
  ierr = BVResize(V,*nc+V->m,PETSC_FALSE);CHKERRQ(ierr);
  ierr = BVInsertVecs(V,0,nc,C,PETSC_TRUE);CHKERRQ(ierr);
  V->nc = *nc;
  V->m  = msave;
  V->ci[0] = -V->nc-1;
  V->ci[1] = -V->nc-1;
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetOptionsPrefix"
/*@C
   BVSetOptionsPrefix - Sets the prefix used for searching for all
   BV options in the database.

   Logically Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
-  prefix - the prefix string to prepend to all BV option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: BVAppendOptionsPrefix(), BVGetOptionsPrefix()
@*/
PetscErrorCode BVSetOptionsPrefix(BV bv,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)bv,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVAppendOptionsPrefix"
/*@C
   BVAppendOptionsPrefix - Appends to the prefix used for searching for all
   BV options in the database.

   Logically Collective on BV

   Input Parameters:
+  bv     - the basis vectors context
-  prefix - the prefix string to prepend to all BV option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: BVSetOptionsPrefix(), BVGetOptionsPrefix()
@*/
PetscErrorCode BVAppendOptionsPrefix(BV bv,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)bv,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVGetOptionsPrefix"
/*@C
   BVGetOptionsPrefix - Gets the prefix used for searching for all
   BV options in the database.

   Not Collective

   Input Parameters:
.  bv - the basis vectors context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

   Note:
   On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: BVSetOptionsPrefix(), BVAppendOptionsPrefix()
@*/
PetscErrorCode BVGetOptionsPrefix(BV bv,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)bv,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVView_Default"
static PetscErrorCode BVView_Default(BV bv,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscInt          j;
  Vec               v;
  PetscViewerFormat format;
  PetscBool         isascii,ismatlab=PETSC_FALSE;
  const char        *bvname,*name;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_MATLAB) ismatlab = PETSC_TRUE;
  }
  if (ismatlab) {
    ierr = PetscObjectGetName((PetscObject)bv,&bvname);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s=[];\n",bvname);CHKERRQ(ierr);
  }
  for (j=-bv->nc;j<bv->m;j++) {
    ierr = BVGetColumn(bv,j,&v);CHKERRQ(ierr);
    ierr = VecView(v,viewer);CHKERRQ(ierr);
    if (ismatlab) {
      ierr = PetscObjectGetName((PetscObject)v,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s=[%s,%s];clear %s\n",bvname,bvname,name,name);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(bv,j,&v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVView"
/*@C
   BVView - Prints the BV data structure.

   Collective on BV

   Input Parameters:
+  bv     - the BV context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode BVView(BV bv,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         isascii;
  PetscViewerFormat format;
  const char        *orthname[2] = {"classical","modified"};
  const char        *refname[3] = {"if needed","never","always"};
  const char        *borthname[2] = {"Gram-Schmidt","Cholesky"};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)bv),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)bv,viewer);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D columns of global length %D\n",bv->m,bv->N);CHKERRQ(ierr);
      if (bv->nc>0) {
        ierr = PetscViewerASCIIPrintf(viewer,"number of constraints: %D\n",bv->nc);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"vector orthogonalization method: %s Gram-Schmidt\n",orthname[bv->orthog_type]);CHKERRQ(ierr);
      switch (bv->orthog_ref) {
        case BV_ORTHOG_REFINE_IFNEEDED:
          ierr = PetscViewerASCIIPrintf(viewer,"orthogonalization refinement: %s (eta: %g)\n",refname[bv->orthog_ref],(double)bv->orthog_eta);CHKERRQ(ierr);
          break;
        case BV_ORTHOG_REFINE_NEVER:
        case BV_ORTHOG_REFINE_ALWAYS:
          ierr = PetscViewerASCIIPrintf(viewer,"orthogonalization refinement: %s\n",refname[bv->orthog_ref]);CHKERRQ(ierr);
          break;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"block orthogonalization method: %s\n",borthname[bv->orthog_block]);CHKERRQ(ierr);
      if (bv->matrix) {
        if (bv->indef) {
          ierr = PetscViewerASCIIPrintf(viewer,"indefinite inner product\n");CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"non-standard inner product\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
        ierr = MatView(bv->matrix,viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      }
      switch (bv->vmm) {
        case BV_MATMULT_VECS:
          ierr = PetscViewerASCIIPrintf(viewer,"doing matmult as matrix-vector products\n");CHKERRQ(ierr);
          break;
        case BV_MATMULT_MAT:
          ierr = PetscViewerASCIIPrintf(viewer,"doing matmult as a single matrix-matrix product\n");CHKERRQ(ierr);
          break;
        case BV_MATMULT_MAT_SAVE:
          ierr = PetscViewerASCIIPrintf(viewer,"doing matmult as a single matrix-matrix product, saving aux matrices\n");CHKERRQ(ierr);
          break;
      }
      if (bv->rrandom) { 
        ierr = PetscViewerASCIIPrintf(viewer,"generating random vectors independent of the number of processes\n");CHKERRQ(ierr);
      }
    } else {
      if (bv->ops->view) { ierr = (*bv->ops->view)(bv,viewer);CHKERRQ(ierr); }
      else { ierr = BVView_Default(bv,viewer);CHKERRQ(ierr); }
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = (*bv->ops->view)(bv,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVRegister"
/*@C
   BVRegister - Adds a new storage format to the BV package.

   Not collective

   Input Parameters:
+  name     - name of a new user-defined BV
-  function - routine to create context

   Notes:
   BVRegister() may be called multiple times to add several user-defined
   basis vectors.

   Level: advanced

.seealso: BVRegisterAll()
@*/
PetscErrorCode BVRegister(const char *name,PetscErrorCode (*function)(BV))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&BVList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVAllocateWork_Private"
PetscErrorCode BVAllocateWork_Private(BV bv,PetscInt s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s>bv->lwork) {
    ierr = PetscFree(bv->work);CHKERRQ(ierr);
    ierr = PetscMalloc1(s,&bv->work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)bv,(s-bv->lwork)*sizeof(PetscScalar));CHKERRQ(ierr);
    bv->lwork = s;
  }
  PetscFunctionReturn(0);
}

