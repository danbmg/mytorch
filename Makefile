##
## my_torch — the programs are Python: "building" them writes small executable
## wrappers that put the repository on sys.path and call the real entry point.
##

GENERATOR	=	my_torch_generator

all:	$(GENERATOR)

$(GENERATOR):
	@printf '%s\n' '#!/usr/bin/env python3' \
		'import os' \
		'import sys' \
		'sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))' \
		'from generator.main import main' \
		'sys.exit(main())' > $@
	@chmod +x $@

tests_run:
	@if python3 -c "import pytest" 2>/dev/null; then \
		python3 -m pytest tests/ -q; \
	else \
		python3 -m unittest discover -s tests -t . ; \
	fi

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@rm -rf .pytest_cache

fclean:	clean
	@rm -f $(GENERATOR)

re:	fclean all

.PHONY:	all clean fclean re tests_run
