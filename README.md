# MSIS AMR

- **Project**: MSIS AMR — Gradio-based UI and model integrations for an Autonomous Mobile Robot (AMR) developed by NPU Team.
- **Language**: Python
- **Key features**: voice + typed command input, map editor & viewer, manual robot control, real-time mapping, camera-based object detection, and model-backed motion/voice/speech components.

**Quick Start**

- Clone or open the repository and install dependencies:

with cuda:
```cmd
conda env create -f env_withCuda.yaml
```

with out cuda:
```cmd
conda env create -f env_withoutCuda.yaml
```

- Run the app (starts the Gradio interface):

```cmd
gradio app.py
```

**Models & Checkpoints**
- For AMR developer team, please download the checkpoints folder at `/CircuitTeam/AMR Project/checkpoints` on NAS server.
- Place `checkpoints` folder inside the `models` folder.
- Contact AMR developer team if you (others) want to get the model checkpoints.
