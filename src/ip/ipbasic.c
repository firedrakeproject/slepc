/*
     Basic routines

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/ipimpl.h"      /*I "slepcip.h" I*/

PetscClassId IP_CLASSID = 0;
PetscLogEvent IP_InnerProduct = 0, IP_Orthogonalize = 0, IP_ApplyMatrix = 0;

#undef __FUNCT__  
#define __FUNCT__ "IPInitializePackage"
/*@C
  IPInitializePackage - This function initializes everything in the IP package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to IPCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode IPInitializePackage(const char *path) 
{
  static PetscBool initialized = PETSC_FALSE;
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Inner product",&IP_CLASSID);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("IPOrthogonalize",IP_CLASSID,&IP_Orthogonalize); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IPInnerProduct",IP_CLASSID,&IP_InnerProduct); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IPApplyMatrix",IP_CLASSID,&IP_ApplyMatrix); CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ip", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(IP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ip", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(IP_CLASSID);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPCreate"
/*@C
   IPCreate - Creates an IP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newip - location to put the IP context

   Level: beginner

   Note: 
   IP objects are not intended for normal users but only for
   advanced user that for instance implement their own solvers.

.seealso: IPDestroy(), IP
@*/
PetscErrorCode IPCreate(MPI_Comm comm,IP *newip)
{
  PetscErrorCode ierr;
  IP ip;

  PetscFunctionBegin;
  PetscValidPointer(newip,2);
  ierr = PetscHeaderCreate(ip,_p_IP,int,IP_CLASSID,-1,"IP",comm,IPDestroy,IPView);CHKERRQ(ierr);
  *newip            = ip;
  ip->orthog_type   = IP_ORTH_CGS;
  ip->orthog_ref    = IP_ORTH_REFINE_IFNEEDED;
  ip->orthog_eta    = 0.7071;
  ip->bilinear_form = IP_INNER_HERMITIAN;
  ip->innerproducts = 0;
  ip->matrix        = PETSC_NULL;
  ip->Bx            = PETSC_NULL;
  ip->xid           = 0;
  ip->xstate        = 0;

  ierr = PetscPublishAll(ip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPSetOptionsPrefix"
/*@C
   IPSetOptionsPrefix - Sets the prefix used for searching for all 
   IP options in the database.

   Collective on IP

   Input Parameters:
+  ip - the innerproduct context
-  prefix - the prefix string to prepend to all IP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: IPAppendOptionsPrefix()
@*/
PetscErrorCode IPSetOptionsPrefix(IP ip,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ip,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPAppendOptionsPrefix"
/*@C
   IPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   IP options in the database.

   Collective on IP

   Input Parameters:
+  ip - the innerproduct context
-  prefix - the prefix string to prepend to all IP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: IPSetOptionsPrefix()
@*/
PetscErrorCode IPAppendOptionsPrefix(IP ip,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ip,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IPGetOptionsPrefix"
/*@C
   IPGetOptionsPrefix - Gets the prefix used for searching for all 
   IP options in the database.

   Not Collective

   Input Parameters:
.  ip - the innerproduct context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: IPSetOptionsPrefix(), IPAppendOptionsPrefix()
@*/
PetscErrorCode IPGetOptionsPrefix(IP ip,const char *prefix[])
{
 PetscErrorCode ierr;
 PetscFunctionBegin;
 PetscValidHeaderSpecific(ip,IP_CLASSID,1);
 PetscValidPointer(prefix,2);
 ierr = PetscObjectGetOptionsPrefix((PetscObject)ip, prefix);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "IPSetFromOptions"
/*@
   IPSetFromOptions - Sets IP options from the options database.

   Collective on IP

   Input Parameters:
.  ip - the innerproduct context

   Notes:  
   To see all options, run your program with the -help option.

   Level: beginner

.seealso: 
@*/
PetscErrorCode IPSetFromOptions(IP ip)
{
  PetscErrorCode ierr;
  const char     *orth_list[2] = { "mgs" , "cgs" };
  const char     *ref_list[3] = { "never" , "ifneeded", "always" };
  PetscReal      r;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  ierr = PetscOptionsBegin(((PetscObject)ip)->comm,((PetscObject)ip)->prefix,"Inner Product (IP) Options","IP");CHKERRQ(ierr);
  i = ip->orthog_type;
  ierr = PetscOptionsEList("-orthog_type","Orthogonalization method","IPSetOrthogonalization",orth_list,2,orth_list[i],&i,PETSC_NULL);CHKERRQ(ierr);
  j = ip->orthog_ref;
  ierr = PetscOptionsEList("-orthog_refinement","Iterative refinement mode during orthogonalization","IPSetOrthogonalization",ref_list,3,ref_list[j],&j,PETSC_NULL);CHKERRQ(ierr);
  r = ip->orthog_eta;
  ierr = PetscOptionsReal("-orthog_eta","Parameter of iterative refinement during orthogonalization","IPSetOrthogonalization",r,&r,PETSC_NULL);CHKERRQ(ierr);
  ierr = IPSetOrthogonalization(ip,(IPOrthogonalizationType)i,(IPOrthogonalizationRefinementType)j,r);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPSetOrthogonalization"
/*@
   IPSetOrthogonalization - Specifies the type of orthogonalization technique
   to be used (classical or modified Gram-Schmidt with or without refinement).

   Collective on IP

   Input Parameters:
+  ip         - the innerproduct context
.  type       - the type of orthogonalization technique
.  refinement - type of refinement
-  eta        - parameter for selective refinement

   Options Database Keys:
+  -orthog_type <type> -  Where <type> is cgs for Classical Gram-Schmidt
                              orthogonalization (default)
                              or mgs for Modified Gram-Schmidt orthogonalization
.  -orthog_refinement <type> -  Where <type> is one of never, ifneeded
                              (default) or always 
-  -orthog_eta <eta> -  For setting the value of eta
    
   Notes:  
   The default settings work well for most problems. 

   The parameter eta should be a real value between 0 and 1 (or PETSC_DEFAULT).
   The value of eta is used only when the refinement type is "ifneeded". 

   When using several processors, MGS is likely to result in bad scalability.

   Level: advanced

.seealso: IPOrthogonalize(), IPGetOrthogonalization(), IPOrthogonalizationType,
          IPOrthogonalizationRefinementType
@*/
PetscErrorCode IPSetOrthogonalization(IP ip,IPOrthogonalizationType type, IPOrthogonalizationRefinementType refinement, PetscReal eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  switch (type) {
    case IP_ORTH_CGS:
    case IP_ORTH_MGS:
      ip->orthog_type = type;
      break;
    default:
      SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
  }
  switch (refinement) {
    case IP_ORTH_REFINE_NEVER:
    case IP_ORTH_REFINE_IFNEEDED:
    case IP_ORTH_REFINE_ALWAYS:
      ip->orthog_ref = refinement;
      break;
    default:
      SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_WRONG,"Unknown refinement type");
  }
  if (eta == PETSC_DEFAULT) {
    ip->orthog_eta = 0.7071;
  } else {
    if (eta <= 0.0 || eta > 1.0) SETERRQ(((PetscObject)ip)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid eta value");    
    ip->orthog_eta = eta;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPGetOrthogonalization"
/*@C
   IPGetOrthogonalization - Gets the orthogonalization settings from the 
   IP object.

   Not Collective

   Input Parameter:
.  ip - inner product context 

   Output Parameter:
+  type       - type of orthogonalization technique
.  refinement - type of refinement
-  eta        - parameter for selective refinement

   Level: advanced

.seealso: IPOrthogonalize(), IPSetOrthogonalization(), IPOrthogonalizationType,
          IPOrthogonalizationRefinementType
@*/
PetscErrorCode IPGetOrthogonalization(IP ip,IPOrthogonalizationType *type,IPOrthogonalizationRefinementType *refinement, PetscReal *eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  if (type) *type = ip->orthog_type;
  if (refinement) *refinement = ip->orthog_ref;
  if (eta) *eta = ip->orthog_eta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPView"
/*@C
   IPView - Prints the IP data structure.

   Collective on IP

   Input Parameters:
+  ip - the innerproduct context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner

.seealso: IPView(), EPSView(), SVDView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode IPView(IP ip,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)ip)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ip,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"IP Object:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  orthogonalization method: ");CHKERRQ(ierr);
    switch (ip->orthog_type) {
      case IP_ORTH_MGS:
        ierr = PetscViewerASCIIPrintf(viewer,"modified Gram-Schmidt\n");CHKERRQ(ierr);
        break;
      case IP_ORTH_CGS:
        ierr = PetscViewerASCIIPrintf(viewer,"classical Gram-Schmidt\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(((PetscObject)ip)->comm,1,"Wrong value of ip->orth_type");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  orthogonalization refinement: ");CHKERRQ(ierr);
    switch (ip->orthog_ref) {
      case IP_ORTH_REFINE_NEVER:
        ierr = PetscViewerASCIIPrintf(viewer,"never\n");CHKERRQ(ierr);
        break;
      case IP_ORTH_REFINE_IFNEEDED:
        ierr = PetscViewerASCIIPrintf(viewer,"if needed (eta: %f)\n",ip->orthog_eta);CHKERRQ(ierr);
        break;
      case IP_ORTH_REFINE_ALWAYS:
        ierr = PetscViewerASCIIPrintf(viewer,"always\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(((PetscObject)ip)->comm,1,"Wrong value of ip->orth_ref");
    }
  } else {
    SETERRQ1(((PetscObject)ip)->comm,1,"Viewer type %s not supported for IP",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPDestroy"
/*@
   IPDestroy - Destroys IP context that was created with IPCreate().

   Collective on IP

   Input Parameter:
.  ip - the inner product context

   Level: beginner

.seealso: IPCreate()
@*/
PetscErrorCode IPDestroy(IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  if (--((PetscObject)ip)->refct > 0) PetscFunctionReturn(0);

  if (ip->matrix) { ierr = MatDestroy(ip->matrix);CHKERRQ(ierr); }
  if (ip->Bx) { ierr = VecDestroy(ip->Bx);CHKERRQ(ierr); }
  ierr = PetscHeaderDestroy(ip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPGetOperationCounters"
/*@
   IPGetOperationCounters - Gets the total number of inner product operations 
   made by the IP object.

   Not Collective

   Input Parameter:
.  ip - the inner product context

   Output Parameter:
.  dots - number of inner product operations
   
   Level: intermediate

.seealso: IPResetOperationCounters()
@*/
PetscErrorCode IPGetOperationCounters(IP ip,PetscInt *dots)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidPointer(dots,2);
  *dots = ip->innerproducts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPResetOperationCounters"
/*@
   IPResetOperationCounters - Resets the counters for inner product operations 
   made by of the IP object.

   Collective on IP

   Input Parameter:
.  ip - the inner product context

   Level: intermediate

.seealso: IPGetOperationCounters()
@*/
PetscErrorCode IPResetOperationCounters(IP ip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  ip->innerproducts = 0;
  PetscFunctionReturn(0);
}

