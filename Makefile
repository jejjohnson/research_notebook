.PHONY: help install update build clear

help:
	@echo "The following make targets are available:"
	@echo "	install		install all dependencies for environment with conda"
	@echo "	update		update all dependencies for environment with conda"
	@echo "	build		build jupyter book"
	@echo " clean 		clean previously built files"

clean:
	conda env create -f environment_jb.yml

update:
	conda env update -f environment_jb.yml --prune

build:
	jupyter-book build research_notebook --all

clean:
	jupyter-book clean research_notebook