# import os
# import streamlit as st
# from streamlit.components.v1 import declare_component

# # compute the absolute path to the `frontend` folder
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _BUILD_DIR = os.path.join(_THIS_DIR, "frontend")

# # register the component
# stt_webspeech = declare_component(
#     name="stt_webspeech",
#     path=_BUILD_DIR,           # <-- now points to components/stt_webspeech/frontend
# )





components/stt_webspeech/__init__.py

import streamlit as st
from pathlib import Path
from streamlit.components.v1 import declare_component

_frontend_dir = Path(__file__).parent / "frontend"

# tell Streamlit where to find our built JS/HTML
_stt_comp = declare_component(
    name="stt_webspeech",
    path=str(_frontend_dir)
)

def transcribe(
    key: str,
    language: str = "en-US",
    continuous: bool = True,
    interim_results: bool = True,
):
    """
    Starts the browser's Web Speech API recognizer
    and returns the live+interim transcript as a string.
    """
    return _stt_comp(
        key=key,
        language=language,
        continuous=continuous,
        interim_results=interim_results,
        default=""
    )













# from pathlib import Path
# import streamlit as st
# rom streamlit.components.v1 import declare_component

# _frontend_dir = Path(__file__).parent / "frontend"
# _component_path = str(_frontend_dir.resolve())

# stt_webspeech = declare_component(
#     name="stt_webspeech",
#     path=_component_path,
# )
