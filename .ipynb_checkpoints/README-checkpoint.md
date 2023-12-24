# ALSIM: Active Learning SIMulator 

**TREC** is a GUI application for video recording and playing. This application 
is especially interesting for new camera devices if they are supported by
Video4Linux.

## Main Features / Comments
Major information:

  - 4 Threads; GUI, Display, Recording, Data acquisition for speed.
  - Multiple resolutions and frame rates through Video4Linux integration
  - Easy to expand for new devices i.e. Dinolite-edge etc.
  - Build with Python,Qt, OpenCV and Numpy
  - Live frame-rate and frame-rate capacity (%) based on thread capacity

## Dependencies
Video4Linux device drivers are needed. To check run in terminal:
```sh
v4l2-ctl
```
If the package is not installed, install it with:
```sh
sudo apt install v4l-utils
```
After installation the visible devices can be checked with:
```sh
v4l2-ctl --list-devices
```
If no devices are visible, the application will not be able to capture any. 
In that case only recorded videos can be played in the application.

## How to get it

Git has to be installed to clone: 
```sh
sudo apt install git
```
Clone the repository to current working directory
```sh
git clone https://github.com/gijsvanhoutum/trec.git
```
We advise to install a new python virtual environment first with:
```sh
python3 -m venv venv
```
Activate environment
```sh
source venv/bin/activate
```
Install all necessary Python packages with:
```sh
pip install -r /trec/requirements.txt
```
## How to run it

To run execute the following from the current working directory:
```sh
python3 /trec/trec/main.py
```

## TODO

- Make dynamic update of menu items possible in GUI. Currently, a newly created
cannot be played back immediately. It requires a complete restart of the app.
- Complete comments for all classes. 
- Explore if completely switching to signals and slots, by replacing the 
queues leads to even better performance.