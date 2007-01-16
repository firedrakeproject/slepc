/*
     Basic routines
*/
#include "src/ip/ipimpl.h"      /*I "slepcip.h" I*/

PetscCookie IP_COOKIE = 0;
PetscEvent IP_InnerProduct = 0, IP_Orthogonalize = 0;

#undef __FUNCT__  
#define __FUNCT__ "EPSInitializePackage"
PetscErrorCode IPInitializePackage(char *path) 
{
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&IP_COOKIE,"Inner product");CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&IP_Orthogonalize,"IPOrthogonalize",IP_COOKIE); CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&IP_InnerProduct,"IPInnerProduct",IP_COOKIE); CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ip", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(IP_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ip", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(IP_COOKIE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPCreate"
PetscErrorCode IPCreate(MPI_Comm comm,IP *newip)
{
  IP ip;

  PetscFunctionBegin;
  PetscValidPointer(newip,2);
  PetscHeaderCreate(ip,_p_IP,struct _IPOps,IP_COOKIE,-1,"IP",comm,IPDestroy,IPView);
  *newip            = ip;
  ip->orthog_type   = IP_CGS_ORTH;
  ip->orthog_ref    = IP_ORTH_REFINE_IFNEEDED;
  ip->orthog_eta    = 0.7071;
  ip->bilinear_form = IPINNER_HERMITIAN;
  ip->innerproducts = 0;
  ip->work[0]       = PETSC_NULL;
  ip->work[1]       = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPSetOptionsPrefix"
PetscErrorCode IPSetOptionsPrefix(IP ip,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ip,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPAppendOptionsPrefix"
PetscErrorCode IPAppendOptionsPrefix(IP ip,const char *prefix)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ip,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPSetFromOptions"
PetscErrorCode IPSetFromOptions(IP ip)
{
  PetscErrorCode ierr;
  const char     *orth_list[3] = { "mgs" , "cgs", "ncgs" };
  const char     *ref_list[3] = { "never" , "ifneeded", "always" };
  PetscReal      r;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  ierr = PetscOptionsBegin(ip->comm,ip->prefix,"Inner Product (IP) Options","IP");CHKERRQ(ierr);
  i = ip->orthog_type;
  ierr = PetscOptionsEList("-orthog_type","Orthogonalization method","IPSetOrthogonalization",orth_list,3,orth_list[i],&i,PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode IPSetOrthogonalization(IP ip,IPOrthogonalizationType type, IPOrthogonalizationRefinementType refinement, PetscReal eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  switch (type) {
    case IP_CGS_ORTH:
    case IP_MGS_ORTH:
      ip->orthog_type = type;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown orthogonalization type");
  }
  switch (refinement) {
    case IP_ORTH_REFINE_NEVER:
    case IP_ORTH_REFINE_IFNEEDED:
    case IP_ORTH_REFINE_ALWAYS:
      ip->orthog_ref = refinement;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown refinement type");
  }
  if (eta == PETSC_DEFAULT) {
    ip->orthog_eta = 0.7071;
  } else {
    if (eta <= 0.0 || eta > 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid eta value");    
    ip->orthog_eta = eta;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPView"
PetscErrorCode IPView(IP ip,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(ip->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(ip,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"IP Object:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  orthogonalization method: ");CHKERRQ(ierr);
    switch (ip->orthog_type) {
      case IP_MGS_ORTH:
        ierr = PetscViewerASCIIPrintf(viewer,"modified Gram-Schmidt\n");CHKERRQ(ierr);
        break;
      case IP_CGS_ORTH:
        ierr = PetscViewerASCIIPrintf(viewer,"classical Gram-Schmidt\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(1,"Wrong value of ip->orth_type");
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
      default: SETERRQ(1,"Wrong value of ip->orth_ref");
    }
  } else {
    SETERRQ1(1,"Viewer type %s not supported for IP",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPDestroy"
PetscErrorCode IPDestroy(IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  if (ip->work[0]) { ierr = VecDestroy(ip->work[0]);CHKERRQ(ierr); }
  if (ip->work[1]) { ierr = VecDestroy(ip->work[1]);CHKERRQ(ierr); }
  if (--ip->refct <= 0) PetscHeaderDestroy(ip);
  PetscFunctionReturn(0);
}
