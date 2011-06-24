classdef SlepcEPS < PetscObject
%
%   SlepcEPS - a SLEPc eigenvalue solver object
%
%   Creation:
%     eps = SlepcEPS;
%     eps.SetType('krylovschur');
%     eps.SetOperators(A,B);
%     (optional) eps.SetProblemType(...);
%     eps.SetFromOptions;
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
    function obj = SlepcEPS(pid,flag)
      if (nargin > 1) 
        %  SelpcEPS(pid,'pobj') uses an already existing SLEPc EPS object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'EPSCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'EPSSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'EPSSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'EPSSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = Solve(obj)
      err = calllib('libslepc', 'EPSSolve', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,A,B)
      if (nargin == 2) 
        err = calllib('libslepc', 'EPSSetOperators', obj.pobj,A.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'EPSSetOperators', obj.pobj,A.pobj,B.pobj);PetscCHKERRQ(err);
      end
    end
    function err = SetProblemType(obj,t)
      err = calllib('libslepc', 'EPSSetProblemType', obj.pobj,t);PetscCHKERRQ(err);
    end
    function [nconv,err] = GetConverged(obj)
      nconv = 0;
      [err,nconv] = calllib('libslepc', 'EPSGetConverged', obj.pobj,nconv);PetscCHKERRQ(err);
    end
    function [lambda,err] = GetEigenvalue(obj,i)
      lambda = 0.0;
      [err,lambda] = calllib('libslepc', 'EPSGetEigenvalue', obj.pobj,i-1,lambda,0);PetscCHKERRQ(err);
    end
    function [relerr,err] = ComputeRelativeError(obj,i)
      relerr = 0.0;
      [err,relerr] = calllib('libslepc', 'EPSComputeRelativeError', obj.pobj,i-1,relerr);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'EPSView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'EPSView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'EPSDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

 
