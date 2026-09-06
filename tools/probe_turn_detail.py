import tempfile
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.paths import DirectoryWorkspace, set_workspace
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry, tool_result

set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
set_config_source(DictConfigSource({'model': {'default': 'fake-model', 'provider': 'openai'}}))
set_credential_source(StaticCredentials('sk-test'))

calls = []
def handler(args, **kw):
    calls.append(args)
    return tool_result(temp_c=21, city=args.get('city'))

registry.register(name='get_weather', toolset='demo',
    schema={'name': 'get_weather', 'description': 'weather',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}},
    handler=handler)

from hermes_core.run_agent import AIAgent
agent = AIAgent(api_key='sk-test', base_url='https://example.invalid/v1', provider='openai',
                model='fake-model', enabled_toolsets=['demo'], quiet_mode=True, max_iterations=5)
script = Script().calls(('get_weather', {'city': 'Rosario'}))
for _ in range(6):
    script.text('Hacen 21 grados en Rosario.')
client = install_fake_client(agent, script)
r = agent.run_conversation('que temperatura hace en Rosario?')
print('=== RESULTADO ===')
print('completed:', r.get('completed'), '| api_calls:', r.get('api_calls'))
print('final:', repr(r.get('final_response'))[:200])
print('tool ejecutada con:', calls)
print('llamadas al modelo:', client.call_count)
for i, req in enumerate(client.requests):
    roles = [m.get('role') for m in req.messages]
    print(f'  req{i}: roles={roles} tools={req.tool_names}')

# Tool failures are wrapped into tool results by design, so they never reach the
# traceback. Print them, or the manifest closer cannot see what is still missing.
for m in client.requests[-1].messages:
    if m.get('role') == 'tool':
        print('  tool result:', m.get('content'))

# The probe only passes when the whole turn worked: the tool ran with the arguments
# the model asked for, its result went back to the model, and the model answered.
assert calls == [{'city': 'Rosario'}], f'tool never ran: {calls}'
assert any(m.get('role') == 'tool' for m in client.requests[-1].messages), 'tool result never reached the model'
assert r.get('completed') is True, r.get('final_response')
print('PROBE OK')
