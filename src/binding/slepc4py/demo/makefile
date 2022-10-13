PYTHON=python

TESTFILES = $(wildcard ex*.py)
TESTS     = $(TESTFILES:.py=)
RUNRULES  = $(addprefix run_,$(TESTS))

.PHONY:test run ${RUNRULES}
test: run

run: ${RUNRULES}

${RUNRULES}:
	${PYTHON} $(subst run_,,$@).py ${SLEPC_OPTIONS}

