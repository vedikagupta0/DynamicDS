export class ApiError extends Error {}

async function req(path: string, opt?: RequestInit) {
  const r = await fetch("/api" + path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(msg);
  }
  return r;
}

export const getJSON = async (path: string) => (await req(path)).json();
export const postJSON = async (path: string, body: unknown) =>
  (await req(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
export const patchJSON = async (path: string, body: unknown) =>
  (await req(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
export const postForm = async (path: string, form: FormData) => req(path, { method: "POST", body: form });
export const postEmpty = (path: string) => req(path, { method: "POST" });
