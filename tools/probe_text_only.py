import tempfile
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.testing import Script, install_fake_client

set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
set_config_source(DictConfigSource({'model': {'default': 'fake-model', 'provider': 'openai'}}))
set_credential_source(StaticCredentials('sk-test'))

from hermes_core.run_agent import AIAgent
agent = AIAgent(api_key='sk-test', base_url='https://example.invalid/v1', provider='openai',
                model='fake-model', enabled_toolsets=[], quiet_mode=True, max_iterations=3)
script = Script()
for _ in range(6):
    script.text('Hola, soy el core.')
client = install_fake_client(agent, script)
r = agent.run_conversation('hola')
print('completed:', r.get('completed'), '| api_calls:', r.get('api_calls'))
print('final:', repr(r.get('final_response'))[:160])
