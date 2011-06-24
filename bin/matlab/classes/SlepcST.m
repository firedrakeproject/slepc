classdef SlepcST < PetscObject
%
%   SlepcST - a SLEPc spectral transformation object
%
%   Creation from an EPS:
%     st = eps.GetST();
%
  properties (Constant)
    HEP=1;
    GHEP=2;
    NHEP=3;
    GNHEP=4;
    PGNHEP=5;
    GHIEP=6;
  end
  methods
    function obj = SlepcST(pid,flag)
      if (nargin > 1) 
        %  SelpcST(pid,'pobj') uses an already existing SLEPc ST object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'STCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'STSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'STSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'STSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,A,B)
      if (nargin == 2) 
        err = calllib('libslepc', 'STSetOperators', obj.pobj,A.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'STSetOperators', obj.pobj,A.pobj,B.pobj);PetscCHKERRQ(err);
      end
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'STView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'STView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function [ksp,err] = GetKSP(obj)
      [err,pid] = calllib('libslepc', 'STGetKSP', obj.pobj,0);PetscCHKERRQ(err);
      ksp = PetscKSP(pid,'pobj');
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'STDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
