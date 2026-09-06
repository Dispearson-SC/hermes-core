import tempfile
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry, tool_result

set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
set_config_source(DictConfigSource({'model': {'default': 'fake-model', 'provider': 'openai'}}))
set_credential_source(StaticCredentials('sk-test'))

registry.register(
    name='get_weather', toolset='demo',
    schema={'name': 'get_weather', 'description': 'weather',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}},
    handler=lambda args, **kw: tool_result(temp_c=21, city=args.get('city')))

from hermes_core.run_agent import AIAgent
agent = AIAgent(api_key='sk-test', base_url='https://example.invalid/v1', provider='openai',
                model='fake-model', enabled_toolsets=['demo'], quiet_mode=True, max_iterations=5)
client = install_fake_client(
    agent, Script().calls(('get_weather', {'city': 'Rosario'})).text('Hacen 21 grados en Rosario.'))
r = agent.run_conversation('que temperatura hace en Rosario?')
print('=== TURNO COMPLETO ===')
print('completed:', r.get('completed'), '| api_calls:', r.get('api_calls'))
print('final:', r.get('final_response'))
print('tool results devueltos al modelo:', client.tool_result_payloads())
