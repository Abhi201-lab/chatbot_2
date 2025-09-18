import importlib
import sys

modules=['chat_api_v1.main','kmapi_v1.main','ingestapi_v1.main','llmapi_v1.main','chatui_v1.main']
failed=[]
for m in modules:
    try:
        importlib.import_module(m)
        print(m,'OK')
    except Exception as e:
        print(m,'FAILED:',e)
        failed.append((m,str(e)))
if failed:
    sys.exit(1)
else:
    sys.exit(0)
