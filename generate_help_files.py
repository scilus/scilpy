import os
import subprocess
from pathlib import Path

# Directory where your scripts are located
scripts_dir = Path('scripts/')

# Hidden directory to store help files
hidden_dir = scripts_dir / '.hidden'
hidden_dir.mkdir(exist_ok=True)

# Iterate over all scripts and generate help files
for script in scripts_dir.glob('*.py'):
	if script.name == '__init__.py' or script.name == 'scil_search_keywords.py':
		continue
	help_file = hidden_dir / f'{script.name}.help'

	# Run the script with --h and capture the output
	result = subprocess.run(['python', script, '--h'], capture_output=True, text=True)

	# Save the output to the hidden file
	with open(help_file, 'w') as f:
		f.write(result.stdout)

	print(f'Help output for {script.name} saved to {help_file}')
