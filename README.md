# ms-e226-project
Stanford's MS&amp;E 226 class project [Pierre-Amaury Laforcade &amp; Axel Durand-Allizé]

## Project setup 

This project uses Poetry to manage its dependencies. To set the project up if you haven't used poetry before, run the following commands, after making sure you use a version of python that's compatible with the project (see pyproject.toml)

```bash
pip install poetry 
poetry install --no-root
```

After that, you need to activate the virtual environment to run the python tasks with the poetry shell. For that, run the following command:
```bash
poetry shell
```

## Run the model tasks

To run the different tasks, you can use the following commands - you put the data folder at the root of the repo

**Preprocessing**
```bash
python core/__main__.py --task preprocessing --data-folder data
```

**Exploration**
```bash
python core/__main__.py --task exploration --data-folder data
```

**Regression**
```bash
python core/__main__.py --task regression --data-folder data
```

**Classification**
```bash
python core/__main__.py --task classification --data-folder data
```