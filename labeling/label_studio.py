import os

# This script starts the Label Studio server.
# Make sure you have Label Studio installed in your environment:
# pip install label-studio

print("Launching Label Studio...")
print("You can access it at http://localhost:8080 once it has started.")

# This command will start Label Studio and it will run in the foreground.
# To stop it, press Ctrl+C in your terminal.
os.system("label-studio start")
