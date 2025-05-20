import os
from dotenv import load_dotenv

import streamlit as st
import urllib.parse
from requests_oauthlib import OAuth2Session

load_dotenv()
GOOGLE_OAUTH_CLIENT_ID= os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET= os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT_URI= os.environ.get("GOOGLE_OAUTH_REDIRECT_URI")

