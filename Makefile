.PHONY: help install_jbook update_jbook install_jaxlib update_jaxlib build clear

help:
	@echo "The following make targets are available:"
	@echo "	install		install all dependencies for environment with conda"
	@echo "	update		update all dependencies for environment with conda"
	@echo "	build		build jupyter book"
	@echo " clean 		clean previously built files"

install_jbook:
	mamba env create -f environment_jb.yml

update_jbook:
	mamba env update -f environment_jb.yml --prune

install_jaxlib:
	mamba env create -f code/jax/environment.yaml

update_jaxlib:
	mamba env update -f code/jax/environment.yaml

build:
	jupyter-book build research_notebook --all

clean:
	jupyter-book clean research_notebook
