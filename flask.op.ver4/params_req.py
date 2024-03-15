from urllib.parse import urlencode
from werkzeug.local import LocalProxy

class BaseResponseParams(object):
    def __init__(self,request,keys=[],strap=""):
        self.session_key = []
        self.request_msg(request,keys)
        self.strap = strap

        try:
            setattr(self,str(request.method).upper(),True)
        except Exception as e:
            print("e: ".e)

    def __getattr__(self, item):
        return ""

    def html_keys(self,name,**kwargs):
        html = f'''name={name} '''
        if getattr(self, name):
            html += f''' value={getattr(self,name)} '''
        html += f''' id=form-field-{name} '''
        html += f''' placeholder={str(name).capitalize()} '''

        for k,v in kwargs.items():
            html +=  f''' {k}={v} '''
        return html

    def request_msg(self,request,session_key=[]):
        keys = []

        if type(request) == dict:
            request = request
        elif type(request) == LocalProxy:
            request = getattr(request, "form" if request.method == "POST" else "args")
        else:
            request = getattr(request,request.method)

        try:
            if not session_key:
                session_key = list(request.keys())
                # print("session_key: ",session_key)
        except Exception as e:
            session_key = []

        for key in session_key:
            val = request.get(key,"").strip()
            if val.isdigit():
                setattr(self, f"{key}_int", int(val))
            setattr(self,key,val)
            keys.append(key)
            # print("===>",key,request.get(key,"").strip())
        self.session_key = keys

    def get_params(self):
        params = {}
        for key in self.session_key:
            params[key] = getattr(self,key)
            # print(key,getattr(self,key))
        return params

    def get_url(self,url,params = None):
        if type(params) == list:
            params = {k:"" for k in params}
        elif type(params) == dict:
            pass
        else:
            params = {}

        for key in params:
            params[key] = getattr(self,key)
        result = urlencode(params)

        url = str(url)
        if url.endswith("?"):
            url = url+result+"&"
        else:
            url = url+"?"+result+"&"
        return url
