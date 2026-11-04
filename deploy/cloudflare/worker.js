// Reverse proxy: stillwater.onefusionlab.systems -> Modal web endpoint.
// Streams request/response bodies (uploads, SSE); adds no logging of bodies.
const ORIGIN = 'https://adribmahmud101--stillwater-web-web.us-east.modal.run';

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = new URL(url.pathname + url.search, ORIGIN);
    const headers = new Headers(request.headers);
    headers.set('X-Forwarded-Host', url.host);
    const init = { method: request.method, headers, redirect: 'manual' };
    if (!['GET', 'HEAD'].includes(request.method)) init.body = request.body;
    return fetch(new Request(target, init));
  },
};
