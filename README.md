# Few-shot and Zero-shot Learning for MIR

## Build the tutorial

```
git clone https://github.com/music-fsl-zsl/tutorial 
cd tutorial
pip install -r requirements.txt
jb build --all book/
```

The book will then be built onto `book/_build`. 

## Serve the tutorial locally

After building the tutorial, you can serve it locally with:

```bash
cd book/_build/html
python -m http.server
```

This will create a local server, which you can access by the port printed out by the command. 

## Contributing to the tutorial

JupyterBook renders markdown files, jupyter notebooks, and markdown notebooks as pages on a book. 

All of the common code required to run the jupyter book is located in the `common/` folder. It is a pip-installable package and can be imported as such. 

All of the content and configuration for the book itself is under `book/`. To configure the book, edit the `_config.yml` file. To edit the table of contents, add or remove new chapters to the book, edit the `_toc.yml` file. 

References are stored under `book/references.bib`, which works just like any other BibTex file. To perform an inline citation, use the `{cite}` keyword in any markdown file or jupyter notebook. To embed the bibliography, use the `{bibliography}` keyword. See [here](https://jupyterbook.org/en/stable/content/citations.html) for more information. 

For more information, see the [JupyterBook documentation](https://jupyterbook.org/en/stable/content/)
