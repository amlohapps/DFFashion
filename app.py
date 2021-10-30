# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 22:05:24 2021

@author: prajw
"""

import os
from flask import Flask, request, redirect, url_for, render_template,flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2

