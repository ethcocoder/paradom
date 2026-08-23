from pathlib import Path

import yaml

path = Path('.github/workflows/upload-to-huggingface.yml')
data = yaml.safe_load(path.read_text())
assert data['name'] == 'Upload completed model artifact to Hugging Face'
assert 'workflow_dispatch' in data[True] or 'workflow_dispatch' in data.get('on', {})
job = data['jobs']['upload']
assert job['env']['SOURCE_RUN_ID'] == '${{ inputs.source_run_id }}'
steps = [step['name'] for step in job['steps']]
assert 'Download completed GitHub artifact' in steps
assert 'Upload model files to Hugging Face' in steps
print('PASS: upload workflow YAML structure and required steps validated.')
