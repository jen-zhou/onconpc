import sys
import os

os.chdir('onconpc_visualization')
sys.path.append('../codes')
print(sys.path)
import utils 
import gradio_utils
import gradio as gr

port = int(os.environ.get('PORT', 8000))
gradio_utils.launch_gradio(server_name='0.0.0.0', server_port=port)