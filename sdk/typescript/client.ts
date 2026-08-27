// TypeScript SDK for Vsector
export class VsectorClient {
  constructor(private baseUrl = "http://localhost:8080/v1", private apiKey?: string) {}
  private headers(): HeadersInit {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["X-API-Key"] = this.apiKey;
    return h;
  }
  async createNamespace(name: string, dimension: number) {
    const r = await fetch(`${this.baseUrl}/namespaces`, { method: "POST", headers: this.headers(), body: JSON.stringify({ name, dimension }) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }
  async upsert(namespace: string, vectors: any[]) {
    const r = await fetch(`${this.baseUrl}/vectors/upsert`, { method: "POST", headers: this.headers(), body: JSON.stringify({ namespace, vectors }) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }
  async query(namespace: string, vector: number[], top_k = 10, filters?: any) {
    const r = await fetch(`${this.baseUrl}/vectors/query`, { method: "POST", headers: this.headers(), body: JSON.stringify({ namespace, vector, top_k, filters }) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }
  async fetch(namespace: string, ids: string[]) {
    const r = await fetch(`${this.baseUrl}/vectors/fetch`, { method: "POST", headers: this.headers(), body: JSON.stringify({ namespace, ids }) });
    return r.json();
  }
}
