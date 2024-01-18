# NatLang AI

NatLang AI is an web application ([www.natlangai.com](https://natlangai.com/)) that applies machine learning to automate quiz creation

The goal of NatLang AI was to build a tool to help students, like me, study. NatLang has been tested with Academic Decathlon coordinators from 5 school districts across DFW and has generated 6000+ questions.

This repository contains code from my production Azure VM, as well as notebooks with code for training my machine learning models and deploying them via Docker containers. A technical documentation overview can be found [here](https://docs.google.com/presentation/d/1s96zr2ARMtuWQZrEMHM7a_eJv26MJaW3mHZMa9yZ-98/edit?usp=sharing).

Here are some key files/directories of note:
* *app.py* contains my code for the web application backend, reference helper modules such as *db.py* and *user.py* for specific tasks such as database management and user authentication
* */static/notebooks/* contains samples of my training notebooks I haved used for the development of my ML language models
* my codebase leverages a variety of open-source libraries which can be found in *requirements.txt*
* */templates* contains HTML and CSS code from the "Landkit – Multipurpose Template" and UI Kit from the Bootstrap Themes catalog, which I have modified to create the webpage layouts for NatLang