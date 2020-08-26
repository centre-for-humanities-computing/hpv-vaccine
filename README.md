# Hpv-Vaccine
_by Marie Louise Tørring, Ane Kathrine Lolholm Gammelby_

## Project Developers
Jan

## Project Description
Tracking usage of topics and shifts of discourse on SoMe.  
(aka per-topic time series + novelty, transience, resonance)

## Prerequisites
Python 3.6 is required for compatability with [guidedlda](https://github.com/vi3k6i5/GuidedLDA).  

```bash
$ sudo pip3 install virtualenv
$ virtualenv -p /usr/bin/python3.6 venv
$ source venv/bin/activate
```

Then you can clone this repository & install requirements to your virtual environment.  
```bash
git clone https://github.com/centre-for-humanities-computing/hpv-vaccine.git
pip3 install -r requirements.txt
```

Preprocessing relies on [text_to_x](https://github.com/centre-for-humanities-computing/text_to_x)  
```bash
pip3 install git+https://github.com/centre-for-humanities-computing/text_to_x
```

## Usage
Right now the analysis is executed from a [driver notebook](https://github.com/centre-for-humanities-computing/hpv-vaccine/blob/master/02_master.ipynb), where you can also find some description of the steps.  

## Data Assessment ##
| Source | risk | Storage | Comment |
| --- | --- | --- | --- |
| FB hpv subset | considerable | Kierkegaard | |
| IM hpv subset | considerable | Kierkegaard | |

## Time Estimate ##

## Deadline ##
October 2020 (first paper)  
TBD (second paper)

## License ##
This software is [MIT licensed](./LICENSE.txt).
