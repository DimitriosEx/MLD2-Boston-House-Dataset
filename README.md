# Boston Data Housing Exercise 
This is an exercise for my [D2] Machine Learning course of Department of Computer Science & Engineering University of Ioannina, UoI.

## Getting Started

### Prerequisites
1. Python (3.6 or higher, preferably 3.9)
2. venv

#### Ubuntu 
```shell
$ sudo apt update && sudo apt upgrade # updates installed packages and repositories metadata
$ sudo apt install python3 python3-pip python3.9-venv python3.9-dev # ubuntu still offers python2 in its repositories
```

### Running the application
1. Create and activate a virtual environment 
    ```shell
   $ python3.9 -m venv venv
   $ source venv/bin/activate
   ```
2. Install necessary python dependencies 
    ```shell
   $ pip install -r requirements.txt # if using pip
   $ poetry install # if using poetry
   ```

3. To run a python file, which is a different implementations/algorithm just run 
    ```shell
    $ python <filename.py>
    ```